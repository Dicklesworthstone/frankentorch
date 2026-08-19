//! Does a `buffer_pool` HIT actually make `avg_pool1d`'s total-coverage lever faster?
//! — `frankentorch-buffer-pool-size-class-fairness-2flpl`, preflighting NEGATIVE_EVIDENCE
//! item 221.
//!
//! WHY THIS EXISTS. Item 221 observed that avg_pool1d's uninit lever fell from 2.37x to
//! 1.28x while the pool's hit rate fell from 72% to 12%, and ATTRIBUTED the first to the
//! second. That attribution is correlation across two binaries, and `recycle()`'s own
//! comment records a REFUTED lever pointing the other way: `frankentorch-7zqbc`
//! implemented eviction, raised the hit rate from 45/134 to 82/134 — "identical in all
//! five invocations, so the mechanism worked exactly as designed" — and **moved no lane's
//! paired ratio at all**.
//!
//! Those two findings cannot both be general. The distinction they turn on is WHICH take
//! API a lane uses:
//!   * `take_zeroed` — a hit saves the ALLOCATION, but the buffer is re-zeroed regardless,
//!     so most of the work survives the hit. 7zqbc's lanes are these.
//!   * `try_take_exact` — a hit is the WHOLE lever: it is the difference between filling
//!     pages already committed and filling pages the kernel must fault in first. Nothing
//!     else about the call changes.
//!
//! So 7zqbc may be right about its lanes and item 221 right about this one, or item 221 may
//! be wrong. This probe decides it directly instead of inferring it from two binaries
//! measured a day apart.
//!
//! WHAT IT IS NOT: this is FT-vs-FT, so it is MAINTENANCE, not a win. It cannot appear in a
//! standing. It exists to test a MECHANISM claim already committed to the ledger.
//!
//! The arms are interleaved HIT/MISS/MISS/HIT within each round so a load ramp lands on both
//! equally, and the pool counters are read per arm so the arms are PROVEN to have differed
//! rather than assumed to.

use std::hint::black_box;
use std::time::Instant;

use ft_core::buffer_pool;

const BATCH: usize = 8;
const CH: usize = 64;
const LEN: usize = 8192;
const KERNEL: usize = 2;
const STRIDE: usize = 2;
const OUT_LEN: usize = LEN / STRIDE;
const NUMEL: usize = BATCH * CH * LEN;

const ROUNDS: usize = 12;

fn one_call() -> Vec<f64> {
    ft_kernel_cpu::avg_pool1d_backward_scalar_f64(1.0, BATCH, CH, LEN, KERNEL, OUT_LEN, STRIDE)
}

fn main() {
    // The lever under test is gated on this being false; if a default ever changes, the
    // probe would silently measure the zeroed path instead and report a null result.
    // Returns the PREVIOUS value, so this both arms the lever and records what it was.
    let previous = ft_kernel_cpu::set_pool_output_zeroed(false);

    println!(
        "probe=pool HIT vs MISS on avg_pool1d_backward_scalar_f64 (FT-vs-FT: MAINTENANCE, not a win)"
    );
    println!(
        "shape=[{BATCH},{CH},{LEN}] kernel={KERNEL} stride={STRIDE} numel={NUMEL} \
         ({:.1} MB per gradient)",
        (NUMEL * 8) as f64 / 1e6
    );
    println!(
        "rayon_threads={} loadavg_1m={:.2}",
        rayon::current_num_threads(),
        ft_api::harness_provenance::load_average_1m().unwrap_or(f64::NAN)
    );

    // Warm: the very first call of the process pays one-off costs that belong to neither arm.
    for _ in 0..3 {
        let g = one_call();
        black_box(g[0]);
        buffer_pool::clear();
    }

    let mut hit_ms: Vec<f64> = Vec::new();
    let mut miss_ms: Vec<f64> = Vec::new();
    let mut churn_ms: Vec<f64> = Vec::new();
    let mut hit_hits = 0u64;
    let mut miss_hits = 0u64;
    let mut churn_hits = 0u64;

    // Arm C exists because arm B may not model the board. On the board a MISS happens with
    // OTHER lanes churning the allocator between calls — `2ca1e43a` added two dense
    // avg_pool1d lanes that allocate a different size in the same sweep. Arm B's allocator
    // sees one size and nothing else, so it can hand the same warm region back every time
    // and a "miss" costs almost nothing. Arm C displaces that region with the forward-output
    // size first, which is what the dense lanes do, so it measures a miss onto pages the
    // allocator has actually let go.
    const CHURN_LEN: usize = BATCH * CH * (LEN / 2); // the pooled forward-output size

    for _ in 0..ROUNDS {
        for &arm in &[0u8, 1, 2, 2, 1, 0] {
            let want_hit = arm == 0;
            if arm == 2 {
                buffer_pool::clear();
                let mut churn = vec![0.0f64; CHURN_LEN];
                // Touch it, or the allocator may never commit the pages and the
                // displacement would not happen.
                churn[0] = 1.0;
                churn[CHURN_LEN - 1] = 1.0;
                black_box(churn[0]);
                drop(churn);
            } else if want_hit {
                // Park an exact-size buffer so `try_take_exact` finds one. Parking a
                // FILLED vec (not a fresh one) matters: its pages are already committed,
                // which is the entire property the lever is claimed to exploit.
                buffer_pool::recycle(vec![0.0f64; NUMEL]);
            } else {
                buffer_pool::clear();
            }

            let before = buffer_pool::stats();
            let started = Instant::now();
            let g = one_call();
            let elapsed = started.elapsed().as_secs_f64() * 1e3;
            black_box(g[0] + g[NUMEL - 1]);
            let after = buffer_pool::stats();
            let got_hit = after.hits - before.hits;

            match arm {
                0 => {
                    hit_ms.push(elapsed);
                    hit_hits += got_hit;
                }
                1 => {
                    miss_ms.push(elapsed);
                    miss_hits += got_hit;
                }
                _ => {
                    churn_ms.push(elapsed);
                    churn_hits += got_hit;
                }
            }
            drop(g);
        }
    }

    let min = |v: &[f64]| v.iter().copied().fold(f64::INFINITY, f64::min);
    let median = |v: &[f64]| {
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        s[s.len() / 2]
    };

    println!();
    println!("  arm    n   min(ms)  median(ms)   pool hits observed");
    println!(
        "  HIT   {:2}   {:7.3}  {:10.3}   {hit_hits} of {}",
        hit_ms.len(),
        min(&hit_ms),
        median(&hit_ms),
        hit_ms.len()
    );
    println!(
        "  MISS  {:2}   {:7.3}  {:10.3}   {miss_hits} of {}",
        miss_ms.len(),
        min(&miss_ms),
        median(&miss_ms),
        miss_ms.len()
    );
    println!(
        "  CHURN {:2}   {:7.3}  {:10.3}   {churn_hits} of {}   (miss + allocator displaced)",
        churn_ms.len(),
        min(&churn_ms),
        median(&churn_ms),
        churn_ms.len()
    );

    // The sentinel: if the arms did not actually differ in pool behaviour, every number
    // above is measuring one thing twice and no conclusion may be drawn from it.
    assert_eq!(
        hit_hits,
        hit_ms.len() as u64,
        "HIT arm did not hit the pool every time — the probe did not test what it claims"
    );
    assert_eq!(
        miss_hits, 0,
        "MISS arm hit the pool — the probe did not test what it claims"
    );
    assert_eq!(
        churn_hits, 0,
        "CHURN arm hit the pool — the probe did not test what it claims"
    );

    println!();
    println!(
        "  min ratio    MISS/HIT = {:.3}x",
        min(&miss_ms) / min(&hit_ms)
    );
    println!(
        "  median ratio MISS/HIT = {:.3}x",
        median(&miss_ms) / median(&hit_ms)
    );
    println!(
        "  min ratio    CHURN/HIT = {:.3}x     median CHURN/HIT = {:.3}x",
        min(&churn_ms) / min(&hit_ms),
        median(&churn_ms) / median(&hit_ms)
    );
    println!();
    println!(
        "  READ: >1 means a pool HIT is faster and item 221's attribution has a mechanism; \
         ~1 means the hit buys nothing here and item 221's causal claim is REFUTED, \
         consistent with 7zqbc."
    );

    ft_kernel_cpu::set_pool_output_zeroed(previous);
}
