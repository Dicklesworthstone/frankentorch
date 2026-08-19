//! Square-SVD forward vs PyTorch with the bidiagonal PARALLEL GATE alternated INSIDE one
//! process — `frankentorch-bidiag-parallel-gate-fork-thrash-mzrnh`.
//!
//! WHAT THE GATE IS. Per Householder reflector, the reduction and the two expansions each run
//! two O(rows x cols) matvecs. Above `PARALLEL_GATE` (`1 << 14`) those matvecs fork across
//! rayon; below it they sweep serially. NEGATIVE_EVIDENCE item 229 measured the fork costing
//! up to 11x on `form_p` alone (n=256: 22.3-37.3 ms gated vs 3.30-3.42 ms with the gate
//! raised), because the fork buys two or three chunks of a matvec that is over in microseconds
//! and pays a rayon wake-up per reflector. Item 232 then enumerated the three call sites and
//! found `form_q` and the reduction's own `dlabrd_panel_f64` were never routed around it.
//!
//! WHY THE ARMS ARE IN ONE PROCESS. The gate used to live in a `OnceLock` read from
//! `FT_LINALG_PARALLEL_GATE`, so an A/B needed one process per arm — a whole launch, a cold
//! allocator and a different window between the two numbers being compared. It is now an
//! atomic (`ft_kernel_cpu::bidiag_parallel_gate_set`), so this lane alternates the arms block
//! by block against ONE incumbent arm in ONE window.
//!
//! ORDERING. For each size the blocks run as a palindrome — arms forward, both PyTorch blocks,
//! arms in reverse — so every FT arm gets one early and one late block and a monotone load
//! ramp lands on all of them equally. The estimator is min-of-`SAMPLES` within a block, then
//! min over each arm's two blocks.
//!
//! WHY THE COUNTS ARE SPLIT BY CALL SITE. NEGATIVE_EVIDENCE item 236 raised
//! `FT_LINALG_PARALLEL_GATE` across six invocations and concluded the gate does not move the SVD
//! forward at n=128-136. That knob reached `reduce_scaled_rows_f64` and `apply_scaled_rank1_f64`
//! but NOT step (12) of `dlabrd_panel_f64`, which tested the bare constant — so its "raised" arm
//! still forked once per reflector inside the reduction, which is 64-70% of the forward at those
//! sizes. The per-site counts are how this lane can say whether that is the reason its own table
//! disagrees with item 236, instead of asserting it.
//!
//! THE BUILT-IN ROUTE CHECK, AND THE ONE THAT DID NOT WORK. Each arm reports how many times a
//! gated call site actually took its parallel branch
//! (`ft_kernel_cpu::bidiag_parallel_branches_take`), so the route is OBSERVED rather than
//! inferred. The first cut of this lane inferred it instead from whether two gate arms produced
//! bit-identical singular values, on the reasoning that `reduce_scaled_rows_f64` is the one
//! branch that moves bits. That inference was WRONG at n=256: the arms were timed 1.32x apart —
//! so they demonstrably ran different code — while their singular-value sums agreed to the last
//! bit, because the QR sweep converges to the same rounded values from slightly different
//! bidiagonal input. Equal output is a necessary condition for the same route, never a
//! sufficient one. The counter cannot go vacuous that way.
//!
//! Run (must be local; rch workers have no PyTorch):
//! ```text
//! RAYON_NUM_THREADS=8 PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
//!   cargo run --release -p frankentorch-api --example bidiag_gate_sweep_h2h
//! ```
//! `FT_GATE_SIZES` (default `128,136,256,512`) and `FT_GATE_VALUES` (default
//! `16384,18446744073709551615`, i.e. the shipped gate and always-serial) select the grid.

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

/// Samples per block; the min of a block is that block's estimate.
const SAMPLES: usize = 5;

/// PyTorch warm-up iterations before ANY timed sample. Four, not two: a cold torch arm reads
/// 5-45x slow on this host and would manufacture a FrankenTorch win out of nothing.
const TORCH_WARMUP: usize = 4;

/// Deterministic and diagonally dominant, built by the SAME closed form on both arms so the
/// singular-value checksum is a real parity check rather than a coincidence of shapes.
fn fill(n: usize) -> Vec<f64> {
    let mut a = vec![0.0_f64; n * n];
    for r in 0..n {
        for c in 0..n {
            let v = ((((r + 2) * (c + 3)) % 17) as f64 - 8.0) * 0.05;
            a[r * n + c] = v + if r == c { 3.0 } else { 0.0 };
        }
    }
    a
}

struct Block {
    ms: f64,
    load: f64,
    mhz: f64,
    iowait: u64,
    /// Sum of the singular values, for parity against the incumbent.
    checksum: f64,
    /// Bit pattern of the checksum. Reported for the arms' agreement, NOT used to infer the
    /// route — see the module comment for why that inference failed.
    checksum_bits: u64,
    /// Parallel branches taken across this whole block, per call site:
    /// `(reduce_scaled_rows, apply_scaled_rank1, dlabrd step 12)`.
    branches: (u64, u64, u64),
    /// FT only: `(reduction, form_p/q expansion, QR sweep)` ns for one instrumented call.
    phases: Option<(u64, u64, u64)>,
}

/// Cumulative iowait jiffies from `/proc/stat`'s aggregate `cpu` line.
fn iowait_jiffies() -> u64 {
    std::fs::read_to_string("/proc/stat")
        .ok()
        .and_then(|text| {
            let line = text.lines().next()?;
            line.split_whitespace().nth(5)?.parse::<u64>().ok()
        })
        .unwrap_or(0)
}

fn provenance() -> (f64, f64) {
    let load = ft_api::harness_provenance::load_average_1m().unwrap_or(f64::NAN);
    let mhz = ft_api::harness_provenance::cpu_mhz_stats()
        .map_or(f64::NAN, |(_min, median, _max, _spread)| median);
    (load, mhz)
}

fn ft_block(n: usize, data: &[f64], gate: u64) -> Block {
    let previous = ft_kernel_cpu::bidiag_parallel_gate_set(gate);
    let _ = ft_kernel_cpu::bidiag_parallel_branches_take();
    let mut best = f64::INFINITY;
    let mut checksum = 0.0;
    let mut checksum_bits = 0u64;
    let iowait_before = iowait_jiffies();
    // One discarded warm sample, matching the incumbent's warm-up in kind.
    for i in 0..=SAMPLES {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(data.to_vec(), vec![n, n], false)
            .expect("svd leaf");
        let started = Instant::now();
        let (_u, sv, _vh) = s.tensor_linalg_svd(x, true).expect("svd");
        let elapsed = started.elapsed().as_secs_f64() * 1e3;
        if i > 0 && elapsed < best {
            best = elapsed;
            let sum: f64 = s.tensor_values(sv).expect("singular values").iter().sum();
            checksum = sum;
            checksum_bits = sum.to_bits();
        }
    }
    // One EXTRA call with the counters cleared, purely to attribute this arm's phases. It is
    // deliberately NOT part of the estimator — mixing an instrumented call into the timed min
    // would report a number nothing else in this campaign is comparable to.
    let _ = ft_kernel_cpu::svd_reduction_sweep_ns_take();
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(data.to_vec(), vec![n, n], false)
            .expect("svd leaf");
        let _ = s.tensor_linalg_svd(x, true).expect("svd");
    }
    let phases = ft_kernel_cpu::svd_reduction_sweep_ns_take();

    let (load, mhz) = provenance();
    let branches = ft_kernel_cpu::bidiag_parallel_branches_take();
    ft_kernel_cpu::bidiag_parallel_gate_set(previous);
    Block {
        ms: best,
        load,
        mhz,
        iowait: iowait_jiffies().saturating_sub(iowait_before),
        checksum,
        checksum_bits,
        branches,
        phases: Some(phases),
    }
}

fn torch_block(python: &str, n: usize, announce: bool) -> Option<Block> {
    let src = format!(
        r#"
import time, torch
torch.set_num_threads(8)
n = {n}
r = torch.arange(n, dtype=torch.float64).reshape(n, 1)
c = torch.arange(n, dtype=torch.float64).reshape(1, n)
A = ((((r + 2) * (c + 3)) % 17) - 8.0) * 0.05 + torch.eye(n, dtype=torch.float64) * 3.0
for _ in range({TORCH_WARMUP}):
    torch.linalg.svd(A)
ts = []
for _ in range({SAMPLES}):
    t = time.perf_counter()
    torch.linalg.svd(A)
    ts.append((time.perf_counter() - t) * 1e3)
_, S, _ = torch.linalg.svd(A)
print("MS", sorted(ts)[0])
print("SSUM", S.sum().item())
print("VER", torch.__version__)
print("THREADS", torch.get_num_threads())
"#
    );
    let iowait_before = iowait_jiffies();
    let out = Command::new(python).arg("-c").arg(&src).output().ok()?;
    if !out.status.success() {
        eprintln!("{}", String::from_utf8_lossy(&out.stderr));
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    let get = |p: &str| {
        text.lines()
            .find_map(|l| l.strip_prefix(p).and_then(|v| v.trim().parse::<f64>().ok()))
    };
    let ms = get("MS ")?;
    let checksum = get("SSUM ")?;
    if announce {
        if let Some(v) = text.lines().find_map(|l| l.strip_prefix("VER ")) {
            let threads = text
                .lines()
                .find_map(|l| l.strip_prefix("THREADS "))
                .unwrap_or("?");
            println!(
                "  incumbent=PyTorch {} threads={} (both self-reported, same invocation)",
                v.trim(),
                threads.trim()
            );
        }
    }
    let (load, mhz) = provenance();
    Some(Block {
        ms,
        load,
        mhz,
        iowait: iowait_jiffies().saturating_sub(iowait_before),
        checksum,
        checksum_bits: checksum.to_bits(),
        branches: (0, 0, 0),
        phases: None,
    })
}

fn gate_label(gate: u64) -> String {
    if gate == u64::MAX {
        "SERIAL".to_string()
    } else {
        format!("{gate}")
    }
}

fn main() {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let sizes: Vec<usize> = std::env::var("FT_GATE_SIZES")
        .unwrap_or_else(|_| "128,136,256,512".to_string())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let gates: Vec<u64> = std::env::var("FT_GATE_VALUES")
        .unwrap_or_else(|_| format!("16384,{}", u64::MAX))
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    assert!(!sizes.is_empty() && !gates.is_empty(), "empty grid");

    println!(
        "measurement=SVD FORWARD ONLY (full U,S,Vh); estimator=min of {SAMPLES} per block, \
         palindrome blocks (arms, PT, PT, arms reversed), min over each arm's two blocks"
    );
    println!(
        "elf_sha256={}",
        ft_api::harness_provenance::executing_elf_sha256()
    );
    println!(
        "rayon_threads={} torch_threads=8 (matched)  default_gate={}",
        rayon::current_num_threads(),
        ft_kernel_cpu::bidiag_parallel_gate()
    );
    println!(
        "arms: bidiag parallel gate in {:?} (u64::MAX = always serial), alternated IN-PROCESS",
        gates.iter().map(|g| gate_label(*g)).collect::<Vec<_>>()
    );
    println!(
        "caveat: no A/A null and block-level (not sample-level) interleaving — weaker than a \
         certified board row, quote as such"
    );
    println!();

    for &n in &sizes {
        let data = fill(n);

        // One DISCARDED block before the palindrome, at this size, on the first arm.
        // NEGATIVE_EVIDENCE item 247 measured the first pass of a sweep at 1.23x the median and
        // 8.13x the worst of a later same-width pass, and it is not contention. Without this the
        // penalty lands on whichever arm the palindrome happens to run first — always arm 0 —
        // which is a systematic bias in favour of every other arm.
        let _ = ft_block(n, &data, gates[0]);

        // Palindrome: arms forward, both incumbent blocks, arms reversed.
        let first: Vec<Block> = gates.iter().map(|&g| ft_block(n, &data, g)).collect();
        let Some(pt_a) = torch_block(&python, n, n == sizes[0]) else {
            eprintln!("n={n}: incumbent arm unavailable; refusing to print a one-armed row");
            continue;
        };
        let Some(pt_b) = torch_block(&python, n, false) else {
            eprintln!("n={n}: incumbent arm unavailable; refusing to print a one-armed row");
            continue;
        };
        let mut second: Vec<Block> = gates.iter().rev().map(|&g| ft_block(n, &data, g)).collect();
        second.reverse();

        let pt_ms = pt_a.ms.min(pt_b.ms);
        println!("n={n}");
        println!(
            "  PT {pt_ms:.3} ms   blocks {:.3}/{:.3}  load {:.2}/{:.2}  MHz {:.0}/{:.0}  \
             iowait {}/{} jiffies",
            pt_a.ms, pt_b.ms, pt_a.load, pt_b.load, pt_a.mhz, pt_b.mhz, pt_a.iowait, pt_b.iowait
        );
        for (idx, &gate) in gates.iter().enumerate() {
            let a = &first[idx];
            let b = &second[idx];
            let ft_ms = a.ms.min(b.ms);
            let ratio = pt_ms / ft_ms;
            let rel = (a.checksum - pt_a.checksum).abs() / (pt_a.checksum.abs() + 1e-300);
            println!(
                "  gate={:<8} FT {ft_ms:.3} ms  {}   blocks {:.3}/{:.3}  spread {:.2}x  \
                 load {:.2}/{:.2}  MHz {:.0}/{:.0}  iowait {}/{}",
                gate_label(gate),
                if ratio >= 1.0 {
                    format!("FT {ratio:.3}x FASTER")
                } else {
                    format!("FT {:.3}x SLOWER", 1.0 / ratio)
                },
                a.ms,
                b.ms,
                a.ms.max(b.ms) / a.ms.min(b.ms),
                a.load,
                b.load,
                a.mhz,
                b.mhz,
                a.iowait,
                b.iowait
            );
            println!(
                "           route: parallel branches (reduce, apply, dlabrd12) block A {:?} \
                 block B {:?}",
                a.branches, b.branches
            );
            if let Some((red, formpq, sweep)) = a.phases {
                let total = (red + formpq + sweep).max(1) as f64;
                println!(
                    "           phases (ours only, one instrumented call): reduction {:.3} ms \
                     {:.0}%  form_p/q {:.3} ms {:.0}%  QR sweep {:.3} ms {:.0}%",
                    red as f64 / 1e6,
                    100.0 * red as f64 / total,
                    formpq as f64 / 1e6,
                    100.0 * formpq as f64 / total,
                    sweep as f64 / 1e6,
                    100.0 * sweep as f64 / total
                );
            }
            println!(
                "           parity singular-value sum: FT {:.12e} PT {:.12e} rel {:.2e} {}",
                a.checksum,
                pt_a.checksum,
                rel,
                if rel < 1e-9 { "MATCH" } else { "MISMATCH" }
            );
        }
        // The route check, from the counter rather than from output bits.
        if gates.len() > 1 {
            let base = first[0].checksum_bits;
            let all_same_bits = first.iter().all(|b| b.checksum_bits == base);
            let counts: Vec<(u64, u64, u64)> = first.iter().map(|b| b.branches).collect();
            let distinct_routes = counts.iter().any(|c| *c != counts[0]);
            println!(
                "  route: parallel-branch counts across gate arms {counts:?} — {}",
                if distinct_routes {
                    "the arms provably took DIFFERENT routes"
                } else {
                    "every arm took the SAME route; a ratio between them prices nothing"
                }
            );
            println!(
                "  arms' singular-value sums are {} (reported, NOT used to infer the route)",
                if all_same_bits {
                    "bit-identical"
                } else {
                    "different"
                }
            );
        }
        println!();
    }
}
