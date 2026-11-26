//! Host copies of title memory placed in a few big heaps that are used
//! again, instead of a committed buffer of their own each
//! (`VICTORY_SHARED_POOL=1`; off by default).
//!
//! Every copy of a window, title buffer or view of title memory
//! (`crate::shadow`) was a committed buffer in the PC's memory the card
//! reads (`Host::shared_buffer`): one kernel allocation, zeroed and mapped,
//! per copy. Copies come and go by the thousand (`crate::shadow` notes
//! 5,263 windows, 4.5 GB, let go and made again in two minutes of run 71),
//! and each creation is time the submitting thread waits.
//!
//! With the switch, a copy of up to a quarter of a block is a buffer placed
//! in one of the pool's heaps (`VICTORY_SHARED_POOL_BLOCK_MB`, 256), at a
//! free range found first-fit on 64 KiB steps ([`crate::ranges`], the
//! bookkeeping texture and staging memory already use). A copy let go hands its
//! buffer to the pool ([`give`]); its range is free again only once nothing
//! but the pool holds the buffer (the lists that used it have finished and
//! let go of it), and memory used before is zeroed when it is handed out
//! again, as a committed buffer's is. Bigger copies, and any the pool
//! cannot place, are committed buffers as before.
//!
//! A report from another fork on the same title (pooled copies, ten-minute
//! run): fewer drops under 8 fps (11 against 15-25) and a median of 15.9
//! against 12.8-14.6 fps; the memory the system counts as shared with the
//! card did not shrink: the copies were not what holds it.

use std::collections::HashMap;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(windows)]
use windows::Win32::Graphics::Direct3D12::*;
#[cfg(windows)]
use windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC;
#[cfg(windows)]
use windows::core::Interface;

/// Placed buffers start on multiples of this (D3D12's default placement
/// alignment), and ranges are cut in steps of it.
const STEP: u64 = 64 << 10;

/// Whether copies come from the pool (`VICTORY_SHARED_POOL=1`). Read once.
pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_SHARED_POOL").is_some_and(|v| v == "1"))
}

/// Bytes of one heap of the pool (`VICTORY_SHARED_POOL_BLOCK_MB`, 256; at
/// least 16).
fn block_bytes() -> u64 {
    static BYTES: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *BYTES.get_or_init(|| {
        let mb = std::env::var("VICTORY_SHARED_POOL_BLOCK_MB")
            .ok()
            .and_then(|v| v.trim().parse::<u64>().ok())
            .unwrap_or(256)
            .max(16);
        mb << 20
    })
}

/// Copies placed, copies too big for a block (committed instead), heaps
/// made and let go, and bytes in use, for the log.
static STATS: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];

#[cfg(windows)]
struct Block {
    heap: ID3D12Heap,
    ranges: crate::ranges::Ranges,
}

#[cfg(windows)]
#[derive(Default)]
struct Pool {
    blocks: Vec<Option<Block>>,
    /// Buffers handed out, by their COM pointer: block, offset, bytes.
    placed: HashMap<usize, (usize, u64, u64)>,
    /// Buffers given back whose range is not free yet: something besides
    /// the pool may still hold them (a list on the card).
    waiting: Vec<(ID3D12Resource, usize, u64, u64)>,
}

#[cfg(windows)]
static POOL: Mutex<Option<Pool>> = Mutex::new(None);

#[cfg(windows)]
impl Pool {
    /// Frees the ranges of buffers given back that nothing else holds now,
    /// and lets go of empty heaps but one.
    fn reclaim(&mut self) {
        let blocks = &mut self.blocks;
        self.waiting.retain(|(buffer, block, at, len)| {
            if crate::host::references(buffer) > 1 {
                return true;
            }
            if let Some(Some(b)) = blocks.get_mut(*block) {
                b.ranges.give(*at, *len);
                STATS[4].fetch_sub(*len, Ordering::Relaxed);
            }
            // The buffer goes with this element, before its range is used again.
            false
        });
        let mut empty_kept = false;
        for slot in self.blocks.iter_mut() {
            if slot
                .as_ref()
                .is_some_and(|b| b.ranges.room() == block_bytes())
            {
                if empty_kept {
                    *slot = None;
                    STATS[3].fetch_add(1, Ordering::Relaxed);
                } else {
                    empty_kept = true;
                }
            }
        }
    }

    /// A new heap, in the first free slot.
    fn grow(&mut self, device: &ID3D12Device) -> Option<usize> {
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: block_bytes(),
            // As `Host::shared_buffer`: the PC's memory, write-back, which
            // the card reads and writes over the bus.
            Properties: D3D12_HEAP_PROPERTIES {
                Type: D3D12_HEAP_TYPE_CUSTOM,
                CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
                MemoryPoolPreference: D3D12_MEMORY_POOL_L0,
                ..Default::default()
            },
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
        };
        let mut made: Option<ID3D12Heap> = None;
        // SAFETY: plain creation; the out pointer is ours.
        if let Err(e) = unsafe { device.CreateHeap::<ID3D12Heap>(&raw const desc, &raw mut made) } {
            tracing::warn!(error = %e, "no heap for the pool of host copies: copies are committed buffers");
            return None;
        }
        let block = Block {
            heap: made?,
            ranges: crate::ranges::Ranges::new(block_bytes()),
        };
        let index = match self.blocks.iter().position(Option::is_none) {
            Some(index) => {
                self.blocks[index] = Some(block);
                index
            }
            None => {
                self.blocks.push(Some(block));
                self.blocks.len() - 1
            }
        };
        let n = STATS[2].fetch_add(1, Ordering::Relaxed) + 1;
        let alive = self.blocks.iter().flatten().count();
        tracing::info!(
            "pool of host copies: heap {n} made ({} MB each); {alive} alive, {} MB in use; {} copies placed and {} too big for a heap so far",
            block_bytes() >> 20,
            STATS[4].load(Ordering::Relaxed) >> 20,
            STATS[0].load(Ordering::Relaxed),
            STATS[1].load(Ordering::Relaxed),
        );
        Some(index)
    }
}

/// A buffer of `size` bytes from the pool, mapped, in the common state,
/// that shaders may read and write: what `Host::shared_buffer` makes, or
/// `None` if the copy is too big for a heap or cannot be placed (the
/// caller then makes a committed one).
#[cfg(windows)]
pub(crate) fn take(device: &ID3D12Device, size: u64) -> Option<(ID3D12Resource, *mut u8)> {
    let len = size.max(1).next_multiple_of(STEP);
    if len > block_bytes() / 4 {
        STATS[1].fetch_add(1, Ordering::Relaxed);
        return None;
    }
    let mut guard = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let pool = guard.get_or_insert_with(Pool::default);
    pool.reclaim();
    let found = pool.blocks.iter_mut().enumerate().find_map(|(i, b)| {
        b.as_mut()
            .and_then(|b| b.ranges.take(len, STEP))
            .map(|(at, used)| (i, at, used))
    });
    let (block, at, used_before) = match found {
        Some(found) => found,
        None => {
            let i = pool.grow(device)?;
            let (at, used) = pool.blocks[i].as_mut()?.ranges.take(len, STEP)?;
            (i, at, used)
        }
    };
    let heap = pool.blocks[block].as_ref()?.heap.clone();
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER,
        Width: len,
        Height: 1,
        DepthOrArraySize: 1,
        MipLevels: 1,
        SampleDesc: DXGI_SAMPLE_DESC {
            Count: 1,
            Quality: 0,
        },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
        Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
        ..Default::default()
    };
    let mut buffer: Option<ID3D12Resource> = None;
    let mut data = std::ptr::null_mut();
    // SAFETY: plain creation in a live heap of ours, at a free range, and
    // mapping of the buffer made.
    let made = unsafe {
        device
            .CreatePlacedResource(
                &heap,
                at,
                &desc,
                D3D12_RESOURCE_STATE_COMMON,
                None,
                &mut buffer,
            )
            .and_then(|()| {
                let b = buffer.as_ref().ok_or_else(windows::core::Error::empty)?;
                b.Map(0, None, Some(&mut data))
            })
    };
    let buffer = match (made, buffer) {
        (Ok(()), Some(buffer)) if !data.is_null() => buffer,
        (made, _) => {
            if let Err(e) = made {
                tracing::warn!(error = %e, size, "no buffer placed in the pool of host copies: a committed one instead");
            }
            if let Some(b) = pool.blocks[block].as_mut() {
                b.ranges.give(at, len);
            }
            return None;
        }
    };
    let data: *mut u8 = data.cast();
    if used_before {
        // SAFETY: the mapped buffer, `len` bytes long and ours alone; a
        // range that held a copy before is zeroed, as a new committed
        // buffer is.
        unsafe { std::ptr::write_bytes(data, 0, len as usize) };
    }
    pool.placed
        .insert(buffer.as_raw() as usize, (block, at, len));
    STATS[0].fetch_add(1, Ordering::Relaxed);
    STATS[4].fetch_add(len, Ordering::Relaxed);
    Some((buffer, data))
}

/// Hands a buffer that [`take`] gave back to the pool when its copy is let
/// go; any other buffer is left alone. Its range is free again once nothing
/// else holds it ([`reclaim`]).
#[cfg(windows)]
pub(crate) fn give(buffer: &ID3D12Resource) {
    let mut guard = POOL.lock().unwrap_or_else(|e| e.into_inner());
    let Some(pool) = guard.as_mut() else {
        return;
    };
    if let Some((block, at, len)) = pool.placed.remove(&(buffer.as_raw() as usize)) {
        pool.waiting.push((buffer.clone(), block, at, len));
    }
}

/// Frees what was given back and is no longer used (once a second, with
/// the sweep, and before each [`take`]).
#[cfg(windows)]
pub(crate) fn reclaim() {
    if let Some(pool) = POOL.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        pool.reclaim();
    }
}
