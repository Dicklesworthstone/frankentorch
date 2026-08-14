//! Bounded recycling pool for large `f64` buffers — `frankentorch-9pafs`.
//!
//! # Why this exists
//!
//! PyTorch ships a caching allocator: a freed tensor's pages stay committed and
//! are handed straight back to the next allocation of that size. FrankenTorch
//! returns every buffer to the system allocator, which for multi-megabyte blocks
//! means `munmap` (or an `madvise` purge under mimalloc) — so the next backward
//! pass re-faults every page it touches. `frankentorch-9pafs` measured that gap
//! at 2.86x on an `avg_pool1d` train step with a process-global caching allocator
//! (`crates/ft-api/examples/pure_rust_caching_alloc_demo.rs`), and the gauntlet
//! scorecard attributes 40-73% of the residual vs-PyTorch loss to it rather than
//! to FrankenTorch's compute.
//!
//! That demo closed the gap by replacing `#[global_allocator]`, which a *library*
//! must never do — the choice is process-global and belongs to the consumer
//! (option C of `frankentorch-1ji9l`). This module captures the same win inside
//! FrankenTorch instead, in 100% safe Rust with no `unsafe` and no global
//! allocator: the buffers that dominate a backward pass are recycled explicitly.
//!
//! # What it is
//!
//! A process-global, bounded free list of `Vec<f64>`, keyed by capacity. It is a
//! *cache*, never a source of truth: [`take_zeroed`] and [`take_filled`] always
//! return a buffer whose contents are exactly what the caller asked for, whether
//! it came from the pool or from a fresh allocation. Callers therefore cannot
//! observe whether a recycle happened — only how long it took.
//!
//! # Bounds
//!
//! Only buffers of at least [`MIN_POOLED_LEN`] elements are parked (small
//! allocations are already cheap and would just add lock traffic), at most
//! [`MAX_PARKED_BUFFERS`] are held, and their total capacity is capped at
//! [`MAX_PARKED_BYTES`]. A recycled buffer that does not fit the budget is simply
//! dropped, so steady-state memory is bounded by that ceiling rather than by the
//! workload.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Smallest buffer worth parking: 32768 f64 = 256 KiB.
///
/// Below this the system allocator serves the request out of an already-committed
/// arena, so there are no page faults to save and parking would only add lock
/// traffic to the hot path.
pub const MIN_POOLED_LEN: usize = 1 << 15;

/// Ceiling on total parked capacity (512 MiB of `f64`).
pub const MAX_PARKED_BYTES: usize = 512 << 20;

/// Ceiling on the number of parked buffers, so [`take_zeroed`]'s best-fit scan
/// stays trivially short.
pub const MAX_PARKED_BUFFERS: usize = 64;

/// A parked buffer is reused for a request only when its capacity is within this
/// factor of the request. Without the bound, one huge parked buffer would satisfy
/// every small request and stay resident forever while the sizes it was meant to
/// serve keep allocating fresh.
const MAX_CAPACITY_SLACK: usize = 4;

static ENABLED: AtomicBool = AtomicBool::new(true);
static PARKED_BYTES: AtomicUsize = AtomicUsize::new(0);
static HITS: AtomicU64 = AtomicU64::new(0);
static MISSES: AtomicU64 = AtomicU64::new(0);
static POOL: Mutex<Vec<Vec<f64>>> = Mutex::new(Vec::new());

/// Pool counters, for tests and for probes that need to prove a lane actually hit
/// the pool rather than assuming it did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PoolStats {
    /// Requests served from a parked buffer.
    pub hits: u64,
    /// Requests that had to allocate.
    pub misses: u64,
    /// Buffers currently parked.
    pub parked_buffers: usize,
    /// Total capacity of parked buffers, in bytes.
    pub parked_bytes: usize,
}

/// Guard against a poisoned lock taking the pool out of service: the pool holds
/// no invariant that a panic could break (it is a list of plain `Vec<f64>`), so
/// the sane recovery is to keep using it.
fn with_pool<R>(f: impl FnOnce(&mut Vec<Vec<f64>>) -> R) -> R {
    let mut guard = match POOL.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    f(&mut guard)
}

/// Whether recycling is currently active.
#[must_use]
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Turn recycling on or off at runtime.
///
/// This exists so a perf harness can A/B the pool inside ONE process against ONE
/// binary — the anchored form this campaign requires, since a two-binary
/// comparison cannot separate the lever from host drift. Turning the pool off
/// does not release what is already parked; call [`clear`] for that.
pub fn set_enabled(enabled: bool) {
    ENABLED.store(enabled, Ordering::Relaxed);
}

/// Current counters.
#[must_use]
pub fn stats() -> PoolStats {
    let parked_buffers = with_pool(|pool| pool.len());
    PoolStats {
        hits: HITS.load(Ordering::Relaxed),
        misses: MISSES.load(Ordering::Relaxed),
        parked_buffers,
        parked_bytes: PARKED_BYTES.load(Ordering::Relaxed),
    }
}

/// Drop every parked buffer and zero the counters.
pub fn clear() {
    with_pool(|pool| {
        pool.clear();
        // Keep this coupled to the list mutation. Otherwise a concurrent
        // recycle can park a buffer between `clear` dropping the lock and this
        // store, leaving the pool nonempty with zero accounted bytes.
        PARKED_BYTES.store(0, Ordering::Relaxed);
    });
    HITS.store(0, Ordering::Relaxed);
    MISSES.store(0, Ordering::Relaxed);
}

/// Take a buffer of `len` elements, every one of them `0.0`.
///
/// Equivalent in every observable way to `vec![0.0; len]`.
#[must_use]
pub fn take_zeroed(len: usize) -> Vec<f64> {
    take_filled(len, 0.0)
}

/// Take a buffer of `len` elements, every one of them `value`.
///
/// Equivalent in every observable way to `vec![value; len]` — including for
/// `-0.0` and NaN payloads, since the fill writes `value` itself rather than
/// relying on a parked buffer's prior contents. A recycled buffer is always
/// cleared and refilled; the pool never returns stale data.
#[must_use]
pub fn take_filled(len: usize, value: f64) -> Vec<f64> {
    if len < MIN_POOLED_LEN || !is_enabled() {
        return vec![value; len];
    }
    let recycled = with_pool(|pool| {
        let mut best: Option<usize> = None;
        for (index, buffer) in pool.iter().enumerate() {
            let capacity = buffer.capacity();
            if capacity < len || capacity > len.saturating_mul(MAX_CAPACITY_SLACK) {
                continue;
            }
            let tighter = match best {
                Some(current) => capacity < pool[current].capacity(),
                None => true,
            };
            if tighter {
                best = Some(index);
            }
        }
        best.map(|index| pool.swap_remove(index))
    });
    match recycled {
        Some(mut buffer) => {
            PARKED_BYTES.fetch_sub(buffer.capacity() * size_of::<f64>(), Ordering::Relaxed);
            HITS.fetch_add(1, Ordering::Relaxed);
            buffer.clear();
            buffer.resize(len, value);
            buffer
        }
        None => {
            MISSES.fetch_add(1, Ordering::Relaxed);
            vec![value; len]
        }
    }
}

/// Build a buffer of `len` elements by writing every one of them.
///
/// This is the pooled counterpart of `ft_kernel_cpu::build_uninit`, for the
/// kernels that already establish they overwrite their whole output: it skips
/// the zero-fill *and* the page faults, where [`take_zeroed`] only skips the
/// faults. On a pool hit the buffer arrives at exactly the right length with its
/// pages already committed, so `fill` does the first touch and nothing else
/// touches the memory at all.
///
/// # Contract
///
/// `fill` MUST write every element of the slice before reading it. A parked
/// buffer arrives holding the PREVIOUS user's values — never uninitialized
/// memory, so a violation is a wrong number rather than undefined behaviour, and
/// this stays inside `#![forbid(unsafe_code)]` — but it is still wrong. Kernels
/// using this must carry a test that parks a poisoned buffer and compares bit
/// patterns against the unpooled result; that is what proves the coverage claim
/// rather than assuming it.
///
/// If the pool cannot serve the request the buffer is freshly zeroed, so a
/// partial `fill` shows up as zeros on a cold pool and as stale values on a warm
/// one. That difference is exactly why the poisoned-buffer test is mandatory.
#[must_use]
pub fn build_overwritten(len: usize, fill: impl FnOnce(&mut [f64])) -> Vec<f64> {
    let mut buffer = take_exact(len);
    fill(&mut buffer);
    buffer
}

/// A buffer of exactly `len` elements whose contents are unspecified but
/// initialized. Private because handing out stale values is only sound practice
/// behind [`build_overwritten`]'s documented contract.
fn take_exact(len: usize) -> Vec<f64> {
    if len < MIN_POOLED_LEN || !is_enabled() {
        return vec![0.0; len];
    }
    // Only an EXACT length match avoids all work: a longer parked buffer would
    // have to be truncated (free) and a shorter one grown (a memset over the new
    // region), and at that point `take_filled` is the honest call.
    let recycled = with_pool(|pool| {
        let index = pool.iter().position(|buffer| buffer.len() == len)?;
        Some(pool.swap_remove(index))
    });
    match recycled {
        Some(buffer) => {
            PARKED_BYTES.fetch_sub(buffer.capacity() * size_of::<f64>(), Ordering::Relaxed);
            HITS.fetch_add(1, Ordering::Relaxed);
            buffer
        }
        None => {
            MISSES.fetch_add(1, Ordering::Relaxed);
            vec![0.0; len]
        }
    }
}

/// Offer a buffer back to the pool.
///
/// Buffers below [`MIN_POOLED_LEN`] capacity, and those that would push the pool
/// past its ceilings, are dropped instead of parked. Always safe to call — the
/// caller gives up ownership either way, exactly as a plain drop would.
///
/// The buffer is parked at its CURRENT length rather than cleared, which is what
/// lets [`build_overwritten`] hand back a right-sized buffer with no fill at all.
/// [`take_filled`] clears and refills regardless, so it cannot observe the
/// difference.
pub fn recycle(buffer: Vec<f64>) {
    let capacity = buffer.capacity();
    if capacity < MIN_POOLED_LEN || !is_enabled() {
        return;
    }
    let bytes = capacity * size_of::<f64>();
    with_pool(|pool| {
        if pool.len() >= MAX_PARKED_BUFFERS
            || PARKED_BYTES.load(Ordering::Relaxed) + bytes > MAX_PARKED_BYTES
        {
            return;
        }
        PARKED_BYTES.fetch_add(bytes, Ordering::Relaxed);
        pool.push(buffer);
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tests share one process-global pool, so they must not run concurrently
    /// against it. Each test takes this lock rather than relying on
    /// `--test-threads=1`.
    static TEST_GUARD: Mutex<()> = Mutex::new(());

    fn guarded<R>(body: impl FnOnce() -> R) -> R {
        let guard = match TEST_GUARD.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        clear();
        set_enabled(true);
        let out = body();
        clear();
        set_enabled(true);
        drop(guard);
        out
    }

    #[test]
    fn take_zeroed_scrubs_a_dirty_recycled_buffer() {
        // THE negative case: an implementation that parks a buffer and hands it
        // back without refilling passes every length check and silently returns
        // the previous pass's gradient. This is the test that fails it.
        guarded(|| {
            let mut dirty = vec![7.5; MIN_POOLED_LEN];
            dirty[0] = f64::NAN;
            recycle(dirty);
            let fresh = take_zeroed(MIN_POOLED_LEN);
            assert_eq!(fresh.len(), MIN_POOLED_LEN);
            assert!(
                fresh.iter().all(|value| *value == 0.0),
                "recycled buffer was handed back without being re-zeroed"
            );
            assert_eq!(stats().hits, 1, "the buffer should have been reused");
        });
    }

    #[test]
    fn build_overwritten_reuses_an_exact_length_buffer_with_no_fill() {
        guarded(|| {
            recycle(vec![7.5; MIN_POOLED_LEN]);
            let built = build_overwritten(MIN_POOLED_LEN, |slice| {
                assert_eq!(slice.len(), MIN_POOLED_LEN);
                // The whole point: what arrives is the PREVIOUS user's data, not
                // zeros — nobody paid to clear it.
                assert!(slice.iter().all(|value| *value == 7.5));
                for (index, value) in slice.iter_mut().enumerate() {
                    *value = index as f64;
                }
            });
            assert_eq!(stats().hits, 1);
            assert_eq!(built.len(), MIN_POOLED_LEN);
            assert!(
                built
                    .iter()
                    .enumerate()
                    .all(|(index, value)| *value == index as f64)
            );
        });
    }

    #[test]
    fn build_overwritten_hands_out_zeros_when_the_pool_is_cold() {
        guarded(|| {
            let built = build_overwritten(MIN_POOLED_LEN, |slice| {
                assert!(
                    slice.iter().all(|value| *value == 0.0),
                    "a cold-pool buffer must be zeroed, so a partial fill degrades \
                     to the old behaviour rather than to garbage"
                );
                slice[0] = 1.0;
            });
            assert_eq!(stats().hits, 0);
            assert_eq!(built[0], 1.0);
            assert!(built[1..].iter().all(|value| *value == 0.0));
        });
    }

    #[test]
    fn build_overwritten_declines_a_mismatched_length() {
        guarded(|| {
            // A buffer of a DIFFERENT length must not be stretched or truncated
            // into service: both cost a memset or a free, which is what
            // `take_filled` is for.
            recycle(vec![1.0; MIN_POOLED_LEN * 2]);
            let built = build_overwritten(MIN_POOLED_LEN, |slice| slice.fill(3.0));
            assert_eq!(built.len(), MIN_POOLED_LEN);
            assert_eq!(stats().hits, 0);
            assert!(built.iter().all(|value| *value == 3.0));
        });
    }

    #[test]
    fn take_filled_scrubs_a_buffer_parked_at_full_length() {
        // `recycle` parks at the buffer's current LENGTH now, so `take_filled`
        // receives a full, dirty buffer rather than an empty one. It must still
        // return exactly the requested fill — this is the interaction between the
        // two take paths, and it is where a length/contents mix-up would hide.
        guarded(|| {
            recycle(vec![f64::NAN; MIN_POOLED_LEN]);
            let taken = take_filled(MIN_POOLED_LEN, 2.5);
            assert_eq!(taken.len(), MIN_POOLED_LEN);
            assert!(taken.iter().all(|value| *value == 2.5));
        });
    }

    #[test]
    fn take_filled_matches_vec_macro_bit_for_bit() {
        guarded(|| {
            for value in [0.0_f64, -0.0, 1.0, f64::NAN, f64::NEG_INFINITY] {
                recycle(vec![3.25; MIN_POOLED_LEN]);
                let pooled = take_filled(MIN_POOLED_LEN, value);
                let plain = vec![value; MIN_POOLED_LEN];
                assert_eq!(pooled.len(), plain.len());
                assert!(
                    pooled
                        .iter()
                        .zip(plain.iter())
                        .all(|(a, b)| a.to_bits() == b.to_bits()),
                    "pooled fill diverged from vec![{value}; n] in the bit pattern"
                );
            }
        });
    }

    #[test]
    fn a_larger_recycled_buffer_is_truncated_to_the_request() {
        guarded(|| {
            let big = vec![1.0; MIN_POOLED_LEN * 2];
            recycle(big);
            let small = take_zeroed(MIN_POOLED_LEN);
            assert_eq!(
                small.len(),
                MIN_POOLED_LEN,
                "length must follow the request, not the parked capacity"
            );
            assert!(small.iter().all(|value| *value == 0.0));
        });
    }

    #[test]
    fn a_too_small_parked_buffer_is_not_reused() {
        guarded(|| {
            recycle(vec![0.0; MIN_POOLED_LEN]);
            let bigger = take_zeroed(MIN_POOLED_LEN * 8);
            assert_eq!(bigger.len(), MIN_POOLED_LEN * 8);
            assert_eq!(
                stats().hits,
                0,
                "a buffer smaller than the request must not be reused"
            );
        });
    }

    #[test]
    fn a_wildly_oversized_parked_buffer_is_not_reused() {
        guarded(|| {
            recycle(vec![0.0; MIN_POOLED_LEN * (MAX_CAPACITY_SLACK + 4)]);
            let request = take_zeroed(MIN_POOLED_LEN);
            assert_eq!(request.len(), MIN_POOLED_LEN);
            assert_eq!(
                stats().hits,
                0,
                "slack bound must keep a huge buffer from serving a small request"
            );
            assert_eq!(
                stats().parked_buffers,
                1,
                "and the oversized buffer stays parked for a request its size"
            );
        });
    }

    #[test]
    fn small_buffers_are_never_parked() {
        guarded(|| {
            recycle(vec![0.0; MIN_POOLED_LEN - 1]);
            assert_eq!(stats().parked_buffers, 0);
            assert_eq!(stats().parked_bytes, 0);
        });
    }

    #[test]
    fn parked_buffer_count_is_bounded() {
        guarded(|| {
            for _ in 0..(MAX_PARKED_BUFFERS + 16) {
                recycle(vec![0.0; MIN_POOLED_LEN]);
            }
            assert_eq!(stats().parked_buffers, MAX_PARKED_BUFFERS);
            assert!(stats().parked_bytes <= MAX_PARKED_BYTES);
        });
    }

    #[test]
    fn parked_bytes_are_bounded() {
        guarded(|| {
            // Each buffer is 1/16 of the ceiling, so the count bound cannot be
            // what stops this — the byte bound has to.
            let len = MAX_PARKED_BYTES / size_of::<f64>() / 16;
            for _ in 0..MAX_PARKED_BUFFERS {
                recycle(vec![0.0; len]);
            }
            assert!(
                stats().parked_bytes <= MAX_PARKED_BYTES,
                "parked {} bytes, ceiling is {MAX_PARKED_BYTES}",
                stats().parked_bytes
            );
            assert!(stats().parked_buffers <= 16);
        });
    }

    #[test]
    fn disabling_the_pool_stops_both_halves() {
        guarded(|| {
            set_enabled(false);
            recycle(vec![0.0; MIN_POOLED_LEN]);
            assert_eq!(
                stats().parked_buffers,
                0,
                "recycle must be a no-op when off"
            );
            let fresh = take_zeroed(MIN_POOLED_LEN);
            assert_eq!(fresh.len(), MIN_POOLED_LEN);
            assert_eq!(stats().hits, 0);
            assert_eq!(stats().misses, 0, "an off pool does not count requests");
        });
    }

    #[test]
    fn parked_bytes_return_to_zero_after_a_full_cycle() {
        guarded(|| {
            recycle(vec![0.0; MIN_POOLED_LEN]);
            let bytes_after_park = stats().parked_bytes;
            assert!(bytes_after_park >= MIN_POOLED_LEN * size_of::<f64>());
            let taken = take_zeroed(MIN_POOLED_LEN);
            assert_eq!(
                stats().parked_bytes,
                0,
                "taking the only parked buffer must decrement the byte accounting"
            );
            drop(taken);
        });
    }

    #[test]
    fn concurrent_take_and_recycle_keep_every_buffer_correct() {
        guarded(|| {
            std::thread::scope(|scope| {
                for thread in 0..8 {
                    scope.spawn(move || {
                        for _ in 0..32 {
                            let len = MIN_POOLED_LEN + thread * 64;
                            let mut buffer = take_filled(len, 2.0);
                            assert_eq!(buffer.len(), len);
                            assert!(buffer.iter().all(|value| *value == 2.0));
                            buffer[0] = 99.0;
                            recycle(buffer);
                        }
                    });
                }
            });
            let stats = stats();
            assert_eq!(stats.hits + stats.misses, 8 * 32);
            assert!(stats.parked_bytes <= MAX_PARKED_BYTES);
        });
    }
}
