//! Square-SVD forward vs PyTorch, straddling the `form_p` blocked/unblocked threshold —
//! `frankentorch-4zjaa`, the vs-incumbent row NEGATIVE_EVIDENCE item 229 owes.
//!
//! WHY THIS EXISTS. Item 229 moved the `bidiag_form_p` dispatch from `n >= 160` to the
//! measured crossover `n >= 130`, and priced the move FT-vs-FT: 1.33-1.51x at n=136 in a
//! quiet window. That is MAINTENANCE evidence under section 1 of the standing orders — it
//! says the dispatch was mis-set, not that we beat the incumbent. This lane is the row that
//! was owed: the same two sizes, against a live PyTorch arm in the SAME invocation.
//!
//! THE TWO SIZES ARE THE POINT. n=128 routes to the unblocked expansion and n=136 to the
//! blocked one, so the pair brackets the threshold. If our standing improves from 128 to 136
//! while PyTorch's does not, the dispatch change is visible from outside our own process —
//! which is the only place a claim about it is worth anything.
//!
//! WHAT THIS IS NOT. There is no A/A null here, and the arms are not interleaved sample by
//! sample the way `gauntlet_lane_sweep_h2h` interleaves them. It is ABBA-balanced in blocks
//! instead: FT, PT, PT, FT, with the min taken over each arm's two blocks, so a monotone
//! load ramp across the invocation lands on both arms equally. A row from here is weaker
//! than a certified board row and must be quoted as such.
//!
//! Run (must be local; rch workers have no PyTorch):
//! ```text
//! RAYON_NUM_THREADS=8 PYTORCH_PYTHON=/path/to/python \
//!   cargo run --release -p frankentorch-api --example svd_square_threshold_h2h
//! ```

use std::process::Command;
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

/// Samples per block. The min of a block is the estimator; two blocks per arm.
const SAMPLES: usize = 5;

/// PyTorch warm-up iterations before ANY timed sample.
///
/// Four, not two: a cold torch arm reads 5-45x slow on this host and would manufacture a
/// FrankenTorch win out of nothing. The FT arm's own first sample is discarded for the same
/// reason.
const TORCH_WARMUP: usize = 4;

/// Deterministic, diagonally dominant so the decomposition is well conditioned, and built by
/// the SAME closed form on both arms so the singular-value checksum is a real parity check
/// rather than a coincidence of shapes.
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
    /// `/proc/stat` iowait jiffies consumed while this block ran. Recorded per arm because
    /// a ratio taken while the host is in iowait is not measuring either implementation.
    iowait: u64,
    checksum: f64,
    /// FT only: `(reduction, form_p/q expansion, QR sweep)` nanoseconds for ONE representative
    /// call, from the counters `ft_kernel_cpu` already maintains. `None` for the incumbent —
    /// PyTorch's LAPACK does not expose its phases, which is exactly why this split can say
    /// where OUR time goes and cannot say where the GAP is.
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

fn ft_block(n: usize, data: &[f64]) -> Block {
    let mut best = f64::INFINITY;
    let mut checksum = 0.0;
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
            checksum = s.tensor_values(sv).expect("singular values").iter().sum();
        }
    }
    // One EXTRA call, with the counters cleared first, purely to attribute this shape's
    // phases. It is deliberately not the timed min sample — mixing an instrumented call into
    // the estimator would report a number nothing else in this campaign is comparable to.
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
    Block {
        ms: best,
        load,
        mhz,
        iowait: iowait_jiffies().saturating_sub(iowait_before),
        checksum,
        phases: Some(phases),
    }
}

fn torch_block(python: &str, n: usize) -> Option<Block> {
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
    if let Some(v) = text.lines().find_map(|l| l.strip_prefix("VER ")) {
        println!(
            "  incumbent=PyTorch {} (self-reported, same invocation)",
            v.trim()
        );
    }
    let (load, mhz) = provenance();
    Some(Block {
        ms,
        load,
        mhz,
        iowait: iowait_jiffies().saturating_sub(iowait_before),
        checksum,
        phases: None,
    })
}

fn main() {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    println!(
        "measurement=SVD FORWARD ONLY (full U,S,Vh); estimator=min of {SAMPLES} per block, \
         ABBA-balanced blocks (FT,PT,PT,FT), min over each arm's two blocks"
    );
    println!(
        "rayon_threads={} torch_threads=8 (matched)",
        rayon::current_num_threads()
    );
    println!(
        "route: n=128 takes the UNBLOCKED form_p expansion, n=136 the BLOCKED one \
         (threshold 130, NEGATIVE_EVIDENCE item 229)"
    );
    println!(
        "caveat: no A/A null and block-level (not sample-level) interleaving — weaker than a \
         certified board row, quote as such"
    );
    println!();

    for &n in &[128usize, 136] {
        let data = fill(n);

        // ABBA in time: A, B, B, A.
        let ft_a = ft_block(n, &data);
        let Some(pt_a) = torch_block(&python, n) else {
            eprintln!("n={n}: incumbent arm unavailable; refusing to print a one-armed row");
            continue;
        };
        let Some(pt_b) = torch_block(&python, n) else {
            eprintln!("n={n}: incumbent arm unavailable; refusing to print a one-armed row");
            continue;
        };
        let ft_b = ft_block(n, &data);

        let ft_ms = ft_a.ms.min(ft_b.ms);
        let pt_ms = pt_a.ms.min(pt_b.ms);
        let ratio = pt_ms / ft_ms;

        // Parity: the singular values are the decomposition's invariant. U and Vh are only
        // determined up to sign/rotation on equal values, so they are NOT compared here.
        let rel = (ft_a.checksum - pt_a.checksum).abs() / (pt_a.checksum.abs() + 1e-300);

        println!(
            "n={n}  FT {ft_ms:.3} ms  PT {pt_ms:.3} ms  {}",
            if ratio >= 1.0 {
                format!("FT {ratio:.2}x FASTER")
            } else {
                format!("FT {:.2}x SLOWER", 1.0 / ratio)
            }
        );
        println!(
            "      FT blocks {:.3}/{:.3} ms  load {:.2}/{:.2}  MHz {:.0}/{:.0}  iowait {}/{} jiffies",
            ft_a.ms, ft_b.ms, ft_a.load, ft_b.load, ft_a.mhz, ft_b.mhz, ft_a.iowait, ft_b.iowait
        );
        if let Some((red, formpq, sweep)) = ft_a.phases {
            let total = (red + formpq + sweep).max(1) as f64;
            println!(
                "      FT phases (ours only, one instrumented call): reduction {:.3} ms {:.0}%  \
                 form_p/q {:.3} ms {:.0}%  QR sweep {:.3} ms {:.0}%",
                red as f64 / 1e6,
                100.0 * red as f64 / total,
                formpq as f64 / 1e6,
                100.0 * formpq as f64 / total,
                sweep as f64 / 1e6,
                100.0 * sweep as f64 / total
            );
        }
        println!(
            "      PT blocks {:.3}/{:.3} ms  load {:.2}/{:.2}  MHz {:.0}/{:.0}  iowait {}/{} jiffies",
            pt_a.ms, pt_b.ms, pt_a.load, pt_b.load, pt_a.mhz, pt_b.mhz, pt_a.iowait, pt_b.iowait
        );
        println!(
            "      parity singular-value sum: FT {:.12e} PT {:.12e} rel {:.2e} {}",
            ft_a.checksum,
            pt_a.checksum,
            rel,
            if rel < 1e-9 { "MATCH" } else { "MISMATCH" }
        );
        // A block spread far wider than the effect means the window moved under the row.
        let ft_spread = ft_a.ms.max(ft_b.ms) / ft_a.ms.min(ft_b.ms);
        let pt_spread = pt_a.ms.max(pt_b.ms) / pt_a.ms.min(pt_b.ms);
        println!("      block spread: FT {ft_spread:.3}x  PT {pt_spread:.3}x");
        println!();
    }
}
