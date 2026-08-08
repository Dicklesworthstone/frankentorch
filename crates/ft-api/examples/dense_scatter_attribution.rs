//! Where does `max_pool3d`'s dense-gradient scatter actually spend its ~1.2 ms?
//! `frankentorch-un3os`.
//!
//! `frankentorch-87sz8` put 84% of the backward in "materialising the dense
//! gradient" and modelled that as contended first-touch page faults on fresh
//! `alloc_zeroed` pages. `frankentorch-zoqws` REFUTED that model in situ: paying
//! the first touch on one thread first made the kernel 1.118x SLOWER and reduced
//! the parallel pass by nothing at all (artifacts/perf/frankentorch-zoqws/).
//!
//! So the term is real but un-modelled. This probe partitions it. The lanes are
//! chosen so that each pair differs in exactly ONE property:
//!
//!   alloc_only              lazy allocation floor, nothing touched
//!   serial_fill             ONE thread writes all 8 MiB
//!   par_fill_64 / _8 / _2   the SAME 8 MiB written by N rayon tasks
//!                           -> isolates task count from everything else
//!   kernel_scatter          the real kernel (parallel, `+=`, f64 offsets)
//!   scatter_serial          the same scatter body on ONE thread
//!                           -> if this beats kernel_scatter, parallelism is the cost
//!   scatter_store           `=` instead of `+=`
//!                           -> isolates the read-for-ownership half of the RMW
//!   scatter_usize_offsets   offsets pre-converted, so no f64->usize per element
//!                           -> isolates the conversion
//!
//! A/A NULL: `alloc_only` is measured twice under two names. Its two medians
//! bound what this harness can resolve; any lane pair closer than that spread is
//! NOT distinguishable here and must not be reported as a difference.
//!
//! Lanes are INTERLEAVED (one rep of every lane, then the next rep) rather than
//! run to completion one at a time, so a load excursion lands on all lanes
//! instead of concentrating in whichever ran during it.
//!
//! FrankenTorch-vs-FrankenTorch attribution only — there is no PyTorch arm here
//! and nothing in this file can produce a vs-upstream ratio.
//!
//! Run: `cargo run --release -p ft-api --features fair-alloc --example dense_scatter_attribution`

use std::hint::black_box;
use std::time::Instant;

use rayon::prelude::*;

const REPS: usize = 25;
const WARMUP: usize = 5;

// The gauntlet's max_pool3d lane.
const N: usize = 2;
const C: usize = 32;
const D: usize = 16;
const H: usize = 32;
const W: usize = 32;
const OD: usize = D / 2;
const OH: usize = H / 2;
const OW: usize = W / 2;

const PLANE_LEN: usize = D * H * W;
const OUT_PLANE_LEN: usize = OD * OH * OW;
const PLANES: usize = N * C;
const DIN_LEN: usize = PLANES * PLANE_LEN;
const DOUT_LEN: usize = PLANES * OUT_PLANE_LEN;

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

/// Scatter body shared by the parallel and serial lanes so they cannot drift.
/// `plane` selects which output plane feeds this input plane.
fn scatter_plane(drow: &mut [f64], plane: usize, arg_offsets: &[f64], dout: &[f64]) {
    let dbase = plane * OUT_PLANE_LEN;
    for i in 0..OUT_PLANE_LEN {
        let oidx = dbase + i;
        let arg = arg_offsets[oidx] as usize;
        drow[arg] += dout[oidx];
    }
}

/// Put the allocator into a KNOWN state before each timed lane: a same-sized
/// block, fully dirtied, then freed.
///
/// Without this the probe measures its own lane ORDER. `vec![0.0; n]` is
/// `alloc_zeroed`, and mimalloc can skip the zeroing when it knows the recycled
/// block is already zero — so a lane that follows a full-8-MiB writer pays a real
/// memset while a lane that follows a one-element toucher pays nothing. The first
/// version of this probe had no conditioner and its A/A pair (two byte-identical
/// `alloc_only` lanes at different positions) came out 0.813 vs 0.138 ms, 83%
/// apart, which vetoed the entire table. Dirtying here makes every lane face the
/// same precondition, and it is also the REALISTIC one: in a training loop the
/// previous iteration's buffers were fully written before being freed.
///
/// Deliberately OUTSIDE the timed region.
fn condition_allocator() {
    let mut w = vec![0.0f64; DIN_LEN];
    w.fill(1.0);
    black_box(&w);
    drop(w);
}

/// Inputs every lane shares. Passed by reference rather than captured, so the
/// lanes need no `'static` bound and the borrow checker stays out of the way.
struct Ctx {
    arg_offsets: Vec<f64>,
    arg_usize: Vec<usize>,
    dout: Vec<f64>,
}

const LANES: [&str; 10] = [
    "alloc_only",
    "alloc_only_AA",
    "serial_fill",
    "par_fill_64",
    "par_fill_8",
    "par_fill_2",
    "kernel_scatter",
    "scatter_serial",
    "scatter_store",
    "scatter_usize_offsets",
];

/// Every lane allocates its own fresh buffer, exactly as the kernel does. Reusing
/// one buffer across reps would hand the fast lanes an allocator-warmth advantage
/// the kernel never gets — the confound that made zoqws's standalone ladder invert.
fn run_lane(idx: usize, ctx: &Ctx) -> f64 {
    match idx {
        // A/A pair: byte-identical bodies under two names.
        0 | 1 => {
            let v = vec![0.0f64; DIN_LEN];
            black_box(&v);
            v[0]
        }
        2 => {
            let mut v = vec![0.0f64; DIN_LEN];
            // 1.0 not 0.0: a fill of zeros over alloc_zeroed memory is a memset over
            // provably-zero memory that LLVM may delete outright, which would make
            // this lane measure nothing. That is precisely the row zoqws's ladder
            // left unverified; here it is un-elidable by construction.
            v.fill(1.0);
            black_box(&v);
            v[0]
        }
        3 | 4 | 5 => {
            let chunk = match idx {
                3 => PLANE_LEN,   // 64 tasks, exactly the kernel's partitioning
                4 => DIN_LEN / 8, // 8 tasks
                _ => DIN_LEN / 2, // 2 tasks
            };
            let mut v = vec![0.0f64; DIN_LEN];
            v.par_chunks_mut(chunk).for_each(|c| c.fill(1.0));
            black_box(&v);
            v[0]
        }
        6 => {
            let v = ft_kernel_cpu::max_pool3d_backward_from_indices_f64(
                &ctx.dout,
                &ctx.arg_offsets,
                N,
                C,
                D,
                H,
                W,
                OD,
                OH,
                OW,
            );
            black_box(&v);
            v[0]
        }
        7 => {
            let mut v = vec![0.0f64; DIN_LEN];
            for (plane, drow) in v.chunks_mut(PLANE_LEN).enumerate() {
                scatter_plane(drow, plane, &ctx.arg_offsets, &ctx.dout);
            }
            black_box(&v);
            v[0]
        }
        8 => {
            let mut v = vec![0.0f64; DIN_LEN];
            v.par_chunks_mut(PLANE_LEN)
                .enumerate()
                .for_each(|(plane, drow)| {
                    let dbase = plane * OUT_PLANE_LEN;
                    for i in 0..OUT_PLANE_LEN {
                        let oidx = dbase + i;
                        // `=` not `+=`. This lane exists only to price the READ half
                        // of the read-modify-write. It is NOT a proposed kernel
                        // change — dropping the accumulate is wrong for overlapping
                        // windows, where two outputs can share an argmax.
                        drow[ctx.arg_offsets[oidx] as usize] = ctx.dout[oidx];
                    }
                });
            black_box(&v);
            v[0]
        }
        _ => {
            let mut v = vec![0.0f64; DIN_LEN];
            v.par_chunks_mut(PLANE_LEN)
                .enumerate()
                .for_each(|(plane, drow)| {
                    let dbase = plane * OUT_PLANE_LEN;
                    for i in 0..OUT_PLANE_LEN {
                        let oidx = dbase + i;
                        drow[ctx.arg_usize[oidx]] += ctx.dout[oidx];
                    }
                });
            black_box(&v);
            v[0]
        }
    }
}

fn main() {
    println!(
        "executing_elf_sha256={}",
        ft_api::harness_provenance::executing_elf_sha256()
    );
    println!(
        "allocator={}",
        if cfg!(feature = "fair-alloc") {
            "mimalloc (--features fair-alloc)"
        } else {
            "system (glibc malloc)"
        }
    );
    println!(
        "rayon_threads={}  load_before={}",
        rayon::current_num_threads(),
        std::fs::read_to_string("/proc/loadavg")
            .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
            .unwrap_or_else(|_| "unknown".to_string())
    );
    println!(
        "din={DIN_LEN} f64 ({} MiB), {PLANES} planes x {PLANE_LEN}, scattered elements={DOUT_LEN} (1 in {})\n",
        DIN_LEN * 8 / (1024 * 1024),
        DIN_LEN / DOUT_LEN
    );

    let input: Vec<f64> = (0..DIN_LEN)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12)
        .collect();
    let (_, arg_offsets) = ft_kernel_cpu::max_pool3d_forward_with_indices_f64(
        &input, N, C, D, H, W, 2, 2, 2, OD, OH, OW, 2, 2, 2,
    );
    let dout = vec![1.0f64; DOUT_LEN];
    let arg_usize: Vec<usize> = arg_offsets.iter().map(|&a| a as usize).collect();

    let ctx = Ctx {
        arg_offsets,
        arg_usize,
        dout,
    };

    let mut samples: Vec<Vec<f64>> = vec![Vec::with_capacity(REPS); LANES.len()];
    for _ in 0..WARMUP {
        for idx in 0..LANES.len() {
            condition_allocator();
            black_box(run_lane(idx, &ctx));
        }
    }
    // Interleaved: one rep of EVERY lane, then the next rep. Running each lane to
    // completion in turn would concentrate any load excursion into whichever lane
    // was unlucky; this spreads it across all of them.
    //
    // The starting lane ROTATES each rep, so no lane permanently owns a position in
    // the cycle. Together with `condition_allocator` that is two independent
    // defences against the ordering confound the A/A pair exists to detect.
    for rep in 0..REPS {
        for step in 0..LANES.len() {
            let idx = (step + rep) % LANES.len();
            condition_allocator();
            let started = Instant::now();
            let guard = run_lane(idx, &ctx);
            let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
            black_box(guard);
            samples[idx].push(elapsed);
        }
    }

    let bytes = (DIN_LEN * 8) as f64;
    println!(
        "{:<24}{:>9}{:>9}{:>9}{:>11}",
        "lane", "median", "min", "max", "GiB/s*"
    );
    for (idx, name) in LANES.iter().enumerate() {
        let s = &samples[idx];
        let m = median(s);
        let gibs = bytes / (m / 1000.0) / (1024.0 * 1024.0 * 1024.0);
        println!(
            "{name:<24}{m:9.3}{:9.3}{:9.3}{gibs:11.2}",
            s.iter().copied().fold(f64::INFINITY, f64::min),
            s.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        );
    }
    println!(
        "\n  *GiB/s counts the 8 MiB output once. Lanes that also READ the output\n   (any `+=`) move roughly twice that, so their GiB/s understates traffic."
    );

    let aa_lo = median(&samples[0]);
    let aa_hi = median(&samples[1]);
    let aa_spread = (aa_hi / aa_lo - 1.0).abs() * 100.0;
    println!(
        "\nA/A NULL: alloc_only={aa_lo:.3} vs alloc_only_AA={aa_hi:.3} -> {aa_spread:.1}% apart.\n\
         Two lanes closer than this are NOT distinguishable by this harness."
    );
    size_sweep();

    println!(
        "\nload_after={}",
        std::fs::read_to_string("/proc/loadavg")
            .map(|s| s.split_whitespace().take(3).collect::<Vec<_>>().join(" "))
            .unwrap_or_else(|_| "unknown".to_string())
    );
}

/// Does serial keep winning as the buffer grows?
///
/// The lane table above says serial beats 64-task parallel at the gauntlet's 8 MiB
/// shape. That alone does NOT license serialising the kernel outright — it licenses
/// serialising it *at that size*. A gate needs to know where, if anywhere, the
/// crossover is. This sweeps the same 1-in-8 scatter shape across buffer sizes and
/// reports the parallel/serial ratio at each; >1.0 means serial wins.
///
/// Same conditioning and interleaving discipline as the main table: the two arms
/// alternate within each rep, and the allocator is conditioned before every timed
/// region.
fn size_sweep() {
    println!("\nSIZE SWEEP — parallel vs serial scatter, same 1-in-8 shape");
    println!(
        "{:>10}{:>12}{:>12}{:>10}",
        "MiB", "par (ms)", "serial (ms)", "par/ser"
    );
    // 1 MiB .. 256 MiB. The gauntlet lane is 8 MiB.
    for &mib in &[1usize, 4, 8, 12, 16, 24, 32, 128, 256] {
        let n = mib * 1024 * 1024 / 8;
        let chunk = (n / 64).max(8);

        let reps = if mib >= 128 { 5 } else { 11 };
        let (mut par, mut ser) = (Vec::with_capacity(reps), Vec::with_capacity(reps));
        for rep in 0..=reps {
            for arm in 0..2 {
                // Alternate which arm leads, as in the two-ELF A/B.
                let serial = (arm + rep) % 2 == 1;
                let mut w = vec![0.0f64; n];
                w.fill(1.0);
                black_box(&w);
                drop(w);

                let started = Instant::now();
                let mut v = vec![0.0f64; n];
                // One scattered element per 64-byte cache line (1-in-8 f64), matching
                // the kernel's density, so every line of the output is touched once.
                if serial {
                    for c in v.chunks_mut(chunk) {
                        for j in 0..c.len() / 8 {
                            c[j * 8] += 1.0;
                        }
                    }
                } else {
                    v.par_chunks_mut(chunk).for_each(|c| {
                        for j in 0..c.len() / 8 {
                            c[j * 8] += 1.0;
                        }
                    });
                }
                let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
                black_box(&v);
                // Discard rep 0: it is the warm-up for this size.
                if rep > 0 {
                    if serial { &mut ser } else { &mut par }.push(elapsed);
                }
            }
        }
        let (mp, ms) = (median(&par), median(&ser));
        println!("{mib:>10}{mp:12.3}{ms:12.3}{:>10.2}", mp / ms);
    }
    println!(
        "  par/ser > 1.0 means the SERIAL arm is faster. NOTE this sweep's inner loop\n\
         scans all offsets per chunk, so its absolute times are not comparable to the\n\
         kernel above — only the par/ser RATIO within a row is meaningful."
    );
}
