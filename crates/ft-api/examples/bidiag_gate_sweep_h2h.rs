//! Square-SVD forward vs PyTorch, with our own arms AND the incumbent alternated ROUND BY ROUND
//! inside one process — `frankentorch-bidiag-parallel-gate-fork-thrash-mzrnh`.
//!
//! WHAT IS MEASURED. `linalg.svd` forward only (full U, S, Vh) on a square matrix, ours against
//! PyTorch, plus as many of our own configurations as the caller asks for. A configuration is a
//! pair: the bidiagonal PARALLEL GATE, and which step-(12) kernel the reduction uses.
//!
//! WHY EVERY ARM IS IN ONE PROCESS. The gate used to live in a `OnceLock` and the step-(12)
//! kernel was a compile-time choice, so an A/B needed one process per arm — a whole launch, a
//! cold allocator, and a different window between the two numbers being compared. Both are now
//! runtime switches (`bidiag_parallel_gate_set`, `bidiag_rowdot_blocked_set`).
//!
//! THE ESTIMATOR IS THE INSTRUMENT (NEGATIVE_EVIDENCE item 255). The first version of this lane
//! timed all of arm A, then all of arm B. Its A/A null — two arms with identical settings, whose
//! difference is therefore pure noise — read 1.02x to 1.19x across four invocations on this host,
//! the size of the effects being chased, and three runs of the same n=512 comparison disagreed on
//! the ordering. Now every arm AND the incumbent are sampled once per ROUND, arm order reversed
//! on odd rounds, first round discarded, and every ratio is the median over rounds of the PAIRED
//! per-round ratio. The null fell to 1.001-1.05x on the same host minutes later. The window was
//! never the binding constraint; the pairing was.
//!
//! WHY THE INCUMBENT IS A CO-PROCESS. Item 256 banked a row whose FT figure was a min over nine
//! rounds and whose PyTorch figure was a min over five samples taken in one block seconds away —
//! an estimator asymmetry biased in our favour, and a gap in time nothing bounded. A child that
//! computes everything and exits cannot be interleaved, so the incumbent is driven as a
//! request/response co-process (`ft_api::harness_interleave`), one timed sample per round, the
//! same warmup count as ours.
//!
//! HOW TO GET A NULL. Repeat an arm in `FT_GATE_VALUES`: two identical arms differ only by the
//! window's own noise, and **no effect smaller than that is readable**. A row whose effect does
//! not clear its own null is unresolved, not a result.
//!
//! THE ROUTE PROOF. Each arm reports how many times a gated call site actually took its parallel
//! branch, split by site. The inference this replaced — that two arms producing bit-identical
//! singular values must have taken the same route — is UNSOUND, and this lane's own data refuted
//! it: at n=256 two arms timed 1.32x apart produced identical singular-value sums, because the QR
//! sweep converges to the same rounded values from slightly different bidiagonal input.
//!
//! Run (must be local; rch workers have no PyTorch):
//! ```text
//! RAYON_NUM_THREADS=8 PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
//!   cargo run --release -p frankentorch-api --example bidiag_gate_sweep_h2h
//! ```
//! `FT_GATE_SIZES` (default `128,136,256,512`), `FT_GATE_VALUES` (default the shipped gate and
//! always-serial), `FT_ROWDOT` (`1` = the four-row step-(12) kernel, `0` = the one-row loop it
//! replaced), `FT_ROUNDS` (default 9) and `FT_H2H_WARMUP` (default 8, read by BOTH arms).

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_api::harness_interleave::{
    QUIT_REQUEST, READY_MARKER, SAMPLE_LOOP_PY, interpreter_args, parse_sample_line, sample_request,
};
use ft_core::ExecutionMode;

/// One measured configuration of our own arm.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Arm {
    gate: u64,
    blocked: bool,
    /// Whether the panel's two trailing updates run as ONE pass over `A22`
    /// (`frankentorch-4zjaa`, NEGATIVE_EVIDENCE item 247b). Bit-identical either way, so this
    /// pair can move time and cannot move a number — `FT_FUSED=1,0` puts both halves in one
    /// invocation against one live incumbent.
    fused: bool,
}

fn arm_label(arm: Arm) -> String {
    let gate = if arm.gate == u64::MAX {
        "SERIAL".to_string()
    } else {
        format!("{}", arm.gate)
    };
    format!(
        "{gate}/{}/{}",
        if arm.blocked { "4row" } else { "1row" },
        if arm.fused { "fused" } else { "2pass" }
    )
}

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

/// One SVD forward under `arm`, in milliseconds, plus the singular-value sum.
///
/// The timer stops before the checksum is read, exactly as the incumbent's `run` does, so both
/// arms time the same region.
fn ft_one(n: usize, data: &[f64], arm: Arm) -> (f64, f64) {
    let previous_gate = ft_kernel_cpu::bidiag_parallel_gate_set(arm.gate);
    let previous_rowdot = ft_kernel_cpu::bidiag_rowdot_blocked_set(arm.blocked);
    let previous_fused = ft_kernel_cpu::bidiag_fused_trailing_set(arm.fused);
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(data.to_vec(), vec![n, n], false)
        .expect("svd leaf");
    let started = Instant::now();
    let (_u, sv, _vh) = session.tensor_linalg_svd(x, true).expect("svd");
    let ms = started.elapsed().as_secs_f64() * 1e3;
    let sum: f64 = session
        .tensor_values(sv)
        .expect("singular values")
        .iter()
        .sum();
    ft_kernel_cpu::bidiag_parallel_gate_set(previous_gate);
    ft_kernel_cpu::bidiag_rowdot_blocked_set(previous_rowdot);
    ft_kernel_cpu::bidiag_fused_trailing_set(previous_fused);
    (ms, sum)
}

/// Ask the incumbent co-process for exactly one timed sample of `lane`.
///
/// A closed stdout is a hard failure rather than a skipped sample: a silently short incumbent arm
/// would leave the remaining rounds measuring only our side, which is the one failure mode a
/// vs-incumbent lane may not have.
fn incumbent_sample(
    stdin: &mut ChildStdin,
    reader: &mut BufReader<ChildStdout>,
    lane: &str,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    writeln!(stdin, "{}", sample_request(lane))?;
    stdin.flush()?;
    let mut line = String::new();
    loop {
        line.clear();
        if reader.read_line(&mut line)? == 0 {
            return Err(format!(
                "the PyTorch co-process closed its stdout while `{lane}` was being sampled; a \
                 partially measured arm cannot carry a vs-PyTorch claim"
            )
            .into());
        }
        if let Some(sample) = parse_sample_line(&line) {
            assert_eq!(sample.lane, lane, "co-process answered for the wrong lane");
            return Ok((sample.milliseconds, sample.gradient_checksum));
        }
    }
}

/// Median of `v`, which is sorted in place. `NaN` for an empty slice.
fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(f64::total_cmp);
    v[v.len() / 2]
}

fn ratio_label(ratio: f64) -> String {
    if ratio >= 1.0 {
        format!("FT {ratio:.3}x FASTER")
    } else {
        format!("FT {:.3}x SLOWER", 1.0 / ratio)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let sizes: Vec<usize> = std::env::var("FT_GATE_SIZES")
        .unwrap_or_else(|_| "128,136,256,512".to_string())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let gate_values: Vec<u64> = std::env::var("FT_GATE_VALUES")
        .unwrap_or_else(|_| format!("262144,{}", u64::MAX))
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let rowdots: Vec<bool> = std::env::var("FT_ROWDOT")
        .unwrap_or_else(|_| "1".to_string())
        .split(',')
        .filter_map(|t| match t.trim() {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        })
        .collect();
    // `frankentorch-4zjaa` item 247b's lever, as a paired arm. Default "1" so an existing command
    // keeps measuring exactly what it measured before; `FT_FUSED=1,0` runs both halves in ONE
    // invocation against ONE live incumbent, which is the only form item 25 admits.
    let fuseds: Vec<bool> = std::env::var("FT_FUSED")
        .unwrap_or_else(|_| "1".to_string())
        .split(',')
        .filter_map(|t| match t.trim() {
            "1" => Some(true),
            "0" => Some(false),
            _ => None,
        })
        .collect();
    assert!(
        !sizes.is_empty() && !gate_values.is_empty() && !rowdots.is_empty() && !fuseds.is_empty(),
        "empty grid"
    );
    let arms: Vec<Arm> = gate_values
        .iter()
        .flat_map(|&gate| {
            rowdots.iter().flat_map(move |&blocked| {
                fuseds
                    .iter()
                    .map(move |&fused| Arm {
                        gate,
                        blocked,
                        fused,
                    })
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    let rounds: usize = std::env::var("FT_ROUNDS")
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(9);
    assert!(rounds >= 1, "FT_ROUNDS must be at least 1");
    // Read by BOTH arms, and matched deliberately: an asymmetric warmup has a bias whose
    // direction depends on which arm is faster, which is a property no instrument may have.
    // Eight rather than the board's 32 because a single n=1024 SVD is ~0.5 s here, and 32 would
    // spend minutes per size before the first sample.
    let warmup: usize = std::env::var("FT_H2H_WARMUP")
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(8);
    // The co-process reads the same count from its OWN environment, handed to it on the
    // `Command` below rather than through `std::env::set_var` — this crate forbids `unsafe`, and
    // mutating the parent's environment to talk to a child is the wrong mechanism anyway.

    let lanes: Vec<(usize, String)> = sizes.iter().map(|&n| (n, format!("svd_{n}"))).collect();
    let lane_entries: Vec<String> = lanes
        .iter()
        .map(|(n, name)| format!("    \"{name}\": (_mk({n}), lambda A: torch.linalg.svd(A)[1]),"))
        .collect();
    let py_setup = format!(
        r#"
import time, torch
torch.set_num_threads(8)
def _mk(n):
    r = torch.arange(n, dtype=torch.float64).reshape(n, 1)
    c = torch.arange(n, dtype=torch.float64).reshape(1, n)
    return ((((r + 2) * (c + 3)) % 17) - 8.0) * 0.05 + torch.eye(n, dtype=torch.float64) * 3.0
def run(base, fn):
    _t = time.perf_counter()
    _s = fn(base)
    _ms = (time.perf_counter() - _t) * 1e3
    return _ms, float(_s.sum())
LANES = {{
{}
}}
print('PT_TORCH_VERSION %s' % torch.__version__, flush=True)
print('PT_THREADS %d' % torch.get_num_threads(), flush=True)
"#,
        lane_entries.join("\n")
    );
    let py = format!("{py_setup}{SAMPLE_LOOP_PY}");

    println!(
        "measurement=SVD FORWARD ONLY (full U,S,Vh); estimator=min over {rounds} rounds, every \
         arm AND the incumbent sampled once per round, arm order reversed on odd rounds, first \
         round discarded; every ratio is the median of the PAIRED per-round ratio"
    );
    println!(
        "elf_sha256={}",
        ft_api::harness_provenance::executing_elf_sha256()
    );
    println!(
        "rayon_threads={} warmup={warmup} (both arms)  default_gate={}",
        rayon::current_num_threads(),
        ft_kernel_cpu::bidiag_parallel_gate()
    );
    println!(
        "arms (gate/step-12 kernel; u64::MAX = always serial, 4row = item 254): {:?}",
        arms.iter().map(|a| arm_label(*a)).collect::<Vec<_>>()
    );
    println!(
        "null: repeat an arm in FT_GATE_VALUES — two identical arms differ only by this window's \
         noise, and no effect below that is readable"
    );

    // `-c`, never `-`: the latter reads the program from stdin until EOF, which deadlocks a
    // co-process whose stdin must stay open for requests.
    let mut child = Command::new(&python)
        .args(interpreter_args(&py))
        .env("FT_H2H_WARMUP", warmup.to_string())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|error| {
            format!(
                "could not start the PyTorch arm (`{python}`): {error}. Set PYTORCH_PYTHON to an \
                 interpreter with torch installed; a FrankenTorch-only run cannot carry a \
                 vs-PyTorch claim."
            )
        })?;
    let mut stdin = child.stdin.take().expect("co-process stdin");
    let mut reader = BufReader::new(child.stdout.take().expect("co-process stdout"));
    // Block until the arm has imported torch, built its tensors and warmed every lane.
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(format!(
                "the PyTorch arm exited before announcing `{READY_MARKER}`; a FrankenTorch-only \
                 run cannot carry a vs-PyTorch claim"
            )
            .into());
        }
        let trimmed = line.trim();
        if let Some(version) = trimmed.strip_prefix("PT_TORCH_VERSION ") {
            println!("incumbent=PyTorch {version} (self-reported, same invocation)");
        }
        if let Some(threads) = trimmed.strip_prefix("PT_THREADS ") {
            println!("incumbent threads={threads} (self-reported)");
        }
        if trimmed == READY_MARKER {
            break;
        }
    }

    for (n, lane) in &lanes {
        let n = *n;
        let data = fill(n);
        let iowait_before = iowait_jiffies();
        let (load_before, mhz_before) = provenance();

        // Matched warmup, ours only: the co-process warmed its lanes before READY.
        for _ in 0..warmup {
            let _ = ft_one(n, &data, arms[0]);
        }

        let mut ft_ms: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); arms.len()];
        let mut ft_sum = vec![0.0f64; arms.len()];
        // ONE INCUMBENT SAMPLE PER ARM PER ROUND, not one per round. With six arms the earlier
        // design ran six of our SVDs against one of theirs, so the incumbent's caches were
        // disturbed six times as often as ours between its own samples — an asymmetry that
        // penalises the incumbent and grows with the number of arms we happen to be sweeping. A
        // measuring instrument whose reading depends on how many of OUR configurations are in the
        // grid is not measuring the incumbent. Each arm is now paired with the incumbent sample
        // that immediately followed it.
        let mut pt_ms: Vec<Vec<f64>> = vec![Vec::with_capacity(rounds); arms.len()];
        let mut pt_sum = 0.0f64;
        for round in 0..=rounds {
            let order: Vec<usize> = if round % 2 == 0 {
                (0..arms.len()).collect()
            } else {
                (0..arms.len()).rev().collect()
            };
            for &idx in &order {
                let (ms, sum) = ft_one(n, &data, arms[idx]);
                let (pt, pt_checksum) = incumbent_sample(&mut stdin, &mut reader, lane)?;
                if round > 0 {
                    ft_ms[idx].push(ms);
                    ft_sum[idx] = sum;
                    pt_ms[idx].push(pt);
                    pt_sum = pt_checksum;
                }
            }
        }
        let pt_all: Vec<f64> = pt_ms.iter().flatten().copied().collect();
        // One extra call per arm, untimed, with the counters cleared: the route AND the phase
        // split. Deliberately NOT part of the estimator — mixing an instrumented call into the
        // timed rounds would report a number nothing else in this campaign is comparable to.
        // THREE instrumented calls per arm, and the phase split is the per-component MEDIAN.
        // NEGATIVE_EVIDENCE item 258c read a single call whose components summed to 1058 ms
        // against a 464 ms median and suspected the counters of double-counting. They do not —
        // the blocked and NR prologues are mutually exclusive and each records once — it was one
        // contended sample. A single sample of anything on this host is not a measurement.
        const PHASE_CALLS: usize = 3;
        let mut branches = vec![(0u64, 0u64, 0u64); arms.len()];
        let mut phases = vec![(0u64, 0u64, 0u64); arms.len()];
        for (idx, &arm) in arms.iter().enumerate() {
            let mut samples: Vec<(u64, u64, u64)> = Vec::with_capacity(PHASE_CALLS);
            for _ in 0..PHASE_CALLS {
                let _ = ft_kernel_cpu::bidiag_parallel_branches_take();
                let _ = ft_kernel_cpu::svd_reduction_sweep_ns_take();
                let _ = ft_one(n, &data, arm);
                branches[idx] = ft_kernel_cpu::bidiag_parallel_branches_take();
                samples.push(ft_kernel_cpu::svd_reduction_sweep_ns_take());
            }
            let mut reduction: Vec<u64> = samples.iter().map(|s| s.0).collect();
            let mut form_pq: Vec<u64> = samples.iter().map(|s| s.1).collect();
            let mut sweep: Vec<u64> = samples.iter().map(|s| s.2).collect();
            reduction.sort_unstable();
            form_pq.sort_unstable();
            sweep.sort_unstable();
            phases[idx] = (
                reduction[PHASE_CALLS / 2],
                form_pq[PHASE_CALLS / 2],
                sweep[PHASE_CALLS / 2],
            );
        }

        let (load_after, mhz_after) = provenance();
        let pt_min = pt_all.iter().copied().fold(f64::INFINITY, f64::min);
        let pt_max = pt_all.iter().copied().fold(0.0f64, f64::max);
        println!();
        println!(
            "n={n}  PT min {pt_min:.3} ms  median {:.3} ms  spread {:.2}x  \
             load {load_before:.2}->{load_after:.2}  MHz {mhz_before:.0}/{mhz_after:.0}  \
             iowait {} jiffies",
            median(&mut pt_all.clone()),
            pt_max / pt_min,
            iowait_jiffies().saturating_sub(iowait_before)
        );
        for (idx, arm) in arms.iter().enumerate() {
            let min = ft_ms[idx].iter().copied().fold(f64::INFINITY, f64::min);
            let med = median(&mut ft_ms[idx].clone());
            let mut vs_pt: Vec<f64> = ft_ms[idx]
                .iter()
                .zip(pt_ms[idx].iter())
                .map(|(ours, theirs)| theirs / ours)
                .collect();
            let mut vs_arm0: Vec<f64> = ft_ms[idx]
                .iter()
                .zip(ft_ms[0].iter())
                .map(|(ours, reference)| reference / ours)
                .collect();
            let rel = (ft_sum[idx] - pt_sum).abs() / (pt_sum.abs() + 1e-300);
            println!(
                "  arm={:<16} min {min:8.3} ms  median {med:8.3} ms  PT-beside-it min \
                 {:8.3} ms  paired-vs-PT {}  paired-vs-arm0 {:.3}x  branches {:?}  \
                 parity rel {rel:.2e} {}",
                arm_label(*arm),
                pt_ms[idx].iter().copied().fold(f64::INFINITY, f64::min),
                ratio_label(median(&mut vs_pt)),
                median(&mut vs_arm0),
                branches[idx],
                if rel < 1e-9 { "MATCH" } else { "MISMATCH" }
            );
            let (reduction, form_pq, sweep) = phases[idx];
            let total = (reduction + form_pq + sweep).max(1) as f64;
            println!(
                "                     phases (ours only, median of 3 instrumented calls): reduction \
                 {:.3} ms {:.0}%  form_p/q {:.3} ms {:.0}%  QR sweep {:.3} ms {:.0}%",
                reduction as f64 / 1e6,
                100.0 * reduction as f64 / total,
                form_pq as f64 / 1e6,
                100.0 * form_pq as f64 / total,
                sweep as f64 / 1e6,
                100.0 * sweep as f64 / total
            );
        }
        // Same gate, different step-(12) kernel, MUST agree bit-for-bit: the four-row kernel
        // preserves each row's own summation order. An assertion rather than a print because it
        // is the only thing standing between an index bug in that kernel and a silently wrong
        // reduction at sizes the golden's small shapes never reach.
        for (i, a) in arms.iter().enumerate() {
            for (j, b) in arms.iter().enumerate() {
                if i < j && a.gate == b.gate && a.blocked != b.blocked {
                    assert_eq!(
                        ft_sum[i].to_bits(),
                        ft_sum[j].to_bits(),
                        "n={n}: {} and {} differ, but the step-(12) kernels are supposed to be \
                         bit-identical",
                        arm_label(*a),
                        arm_label(*b)
                    );
                }
            }
        }
    }

    writeln!(stdin, "{QUIT_REQUEST}")?;
    stdin.flush()?;
    drop(stdin);
    let _ = child.wait();
    Ok(())
}
