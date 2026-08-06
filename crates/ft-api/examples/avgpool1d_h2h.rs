//! `avg_pool1d` sum-loss train step (f64), FrankenTorch vs a live PyTorch arm in the
//! same invocation — `frankentorch-ujw3g`.
//!
//! WHY THIS EXISTS. The `avg_pool1d` gap is the last genuine vs-PyTorch loss on the
//! re-baselined list (`artifacts/perf/frankentorch-ug4ep/`), but the canonical
//! harness for it — `pytorch_gauntlet_bench` — **cannot be run through `rch`**:
//! the workers have no PyTorch, so the `pytorch_2_12_cpu` arm dies with
//! `benchmark failed with status Some(1)`, and `rch` does not sync bench binaries
//! back to the local box either (it syncs examples, not `deps/`). That leaves the
//! FrankenTorch arms measurable but no incumbent to measure them against, which is
//! exactly the "no live incumbent in the same invocation" failure the campaign's
//! evidence contract forbids. An example runs locally where PyTorch lives and
//! syncs back, so this is the shape that can actually carry the claim.
//!
//! It also buys something the criterion lane does not have: an A/A null gate, a
//! landed-win anchor, and an in-process ELF SHA-256.
//!
//! Run: `PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example avgpool1d_h2h`
//! Quote it under `--features fair-alloc`; the default allocator inflates the
//! FrankenTorch side of any lane that rebuilds a large input per iteration.

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

/// The gauntlet's `avg_pool1d` shape, so this harness and that lane describe the
/// same workload.
const N: usize = 8;
const C: usize = 64;
const L: usize = 8192;
const KERNEL: usize = 2;
const STRIDE: usize = 2;

/// Anchor shape: a `cat` big enough to be worth timing, used only as a
/// known-value sanity check on the measurement environment.
const ANCHOR: usize = 4000;

const REPS: usize = 21;
const BOOTSTRAP_REPS: usize = 2_000;

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    values[values.len() / 2]
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

fn median_ratio_ci(numerator: &[f64], denominator: &[f64]) -> (f64, f64, f64) {
    assert_eq!(numerator.len(), denominator.len());
    let point = median(numerator.to_vec()) / median(denominator.to_vec());
    let mut samples = Vec::with_capacity(BOOTSTRAP_REPS);
    let mut state = 0x51a3_7c9d_0e26_b481_u64;
    for _ in 0..BOOTSTRAP_REPS {
        let mut left = Vec::with_capacity(numerator.len());
        let mut right = Vec::with_capacity(denominator.len());
        for _ in 0..numerator.len() {
            let index = (next_random(&mut state) as usize) % numerator.len();
            left.push(numerator[index]);
            right.push(denominator[index]);
        }
        samples.push(median(left) / median(right));
    }
    samples.sort_by(f64::total_cmp);
    (
        point,
        samples[BOOTSTRAP_REPS * 25 / 1_000],
        samples[BOOTSTRAP_REPS * 975 / 1_000],
    )
}

fn executable_sha256() -> String {
    let executable = std::env::current_exe().expect("current executable must be available");
    let output = Command::new("sha256sum")
        .arg(executable)
        .output()
        .expect("sha256sum must be available");
    assert!(output.status.success(), "sha256sum failed");
    String::from_utf8(output.stdout)
        .expect("sha256sum output must be UTF-8")
        .split_whitespace()
        .next()
        .expect("sha256sum must print a digest")
        .to_owned()
}

fn input_values() -> Vec<f64> {
    (0..N * C * L)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12)
        .collect()
}

/// One full train step: materialise the leaf, pool, sum, backward. This mirrors
/// what `benches/pytorch_avg_pool1d_grad.py` does on the PyTorch side, which
/// likewise rebuilds its input inside the loop — both arms pay for setup, and the
/// point of the comparison is what that setup costs on each side.
fn frankentorch_step(base: &[f64]) -> (f64, f64) {
    let started = Instant::now();
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(base.to_vec(), vec![N, C, L], true)
        .expect("leaf");
    let pooled = session
        .functional_avg_pool1d(x, KERNEL, STRIDE)
        .expect("avg_pool1d");
    let loss = session.tensor_sum(pooled).expect("sum");
    let report = session.tensor_backward(loss).expect("backward");
    let checksum = report.gradient(x).expect("grad").iter().sum::<f64>();
    (started.elapsed().as_secs_f64() * 1_000.0, checksum)
}

/// The same train step through the **fused** scalar-loss API. This is the arm the
/// gauntlet calls `frankentorch_kgs4_134_fused_sum_loss`, and it is the ceiling of
/// the "route `tensor_sum(avg_pool1d(x))` to the fused path automatically" lever:
/// no routing change can beat calling the fused API directly. Measuring it beside
/// the compose says whether that lever is worth building at all.
fn frankentorch_fused_step(base: &[f64]) -> (f64, f64) {
    let started = Instant::now();
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = session
        .tensor_variable(base.to_vec(), vec![N, C, L], true)
        .expect("leaf");
    let loss = session
        .functional_avg_pool1d_sum(x, KERNEL, STRIDE)
        .expect("avg_pool1d_sum");
    let report = session.tensor_backward(loss).expect("backward");
    let checksum = report.gradient(x).expect("grad").iter().sum::<f64>();
    (started.elapsed().as_secs_f64() * 1_000.0, checksum)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let base = input_values();
    let anchor_values: Vec<f64> = (0..ANCHOR * ANCHOR).map(|i| (i % 17) as f64).collect();

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
import torch.nn.functional as Fn
torch.set_num_threads(8)
N,C,L,K,S,A=8,64,8192,2,2,4000
base=((torch.arange(N*C*L,dtype=torch.int64)%251).double())*0.001-0.12
base=base.reshape(N,C,L)
m=((torch.arange(A*A,dtype=torch.int64)%17).double()).reshape(A,A)
def step():
    x=base.detach().clone().requires_grad_(True)
    y=Fn.avg_pool1d(x,K,S)
    y.sum().backward()
    return x.grad
def t(fn,n=7):
    for _ in range(4):
        try: fn()
        except Exception: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT cat_anchor %.4f"%t(lambda: torch.cat([m,m],1)))
print("PT avg_pool1d %.4f"%t(step))
g=step()
print("PT grad_sum %.12g"%g.sum().item())
print("PT grad_probe",*("%.12g"%g.flatten()[i].item() for i in (0,1,4095,262143)))
"#;
    let mut child = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    child
        .stdin
        .as_mut()
        .ok_or_else(|| std::io::Error::other("no stdin"))?
        .write_all(py.as_bytes())?;
    let out = child.wait_with_output();
    let pt = out
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default();
    assert!(
        !pt.is_empty(),
        "the PyTorch arm must run in this same invocation; set PYTORCH_PYTHON to an \
         interpreter with torch installed. A FrankenTorch-only run cannot carry a \
         vs-PyTorch claim."
    );
    let pt_value = |name: &str| -> Option<f64> {
        pt.lines().find_map(|line| {
            let mut it = line.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == name {
                it.next()?.parse::<f64>().ok()
            } else {
                None
            }
        })
    };

    // ── parity first: a timing lane is worthless if the two sides compute
    // different things, and nothing else here checks that.
    let (_, checksum) = frankentorch_step(&base);
    let python_grad_sum = pt_value("grad_sum").expect("PyTorch grad_sum missing");
    let tolerance = 1e-9 * python_grad_sum.abs().max(1.0);
    assert!(
        (checksum - python_grad_sum).abs() <= tolerance,
        "avg_pool1d gradient sum disagrees with PyTorch: FT {checksum}, PT {python_grad_sum}, \
         tolerance {tolerance}"
    );
    println!("  avg_pool1d f64 parity: gradient sum matches PyTorch ({checksum:.12e})");

    // The fused API must agree with the compose bit-for-bit, or the routing lever
    // it represents would change results and is off the table regardless of speed.
    let (_, fused_checksum) = frankentorch_fused_step(&base);
    assert_eq!(
        checksum.to_bits(),
        fused_checksum.to_bits(),
        "functional_avg_pool1d_sum must be bit-identical to sum(avg_pool1d(x)); \
         compose {checksum}, fused {fused_checksum}"
    );
    println!("  fused vs compose: gradient sums are bit-identical");

    for _ in 0..4 {
        std::hint::black_box(frankentorch_step(&base));
        std::hint::black_box(frankentorch_fused_step(&base));
    }

    // ── A/A null gate: the same arm against itself must come out at 1.0. If it
    // does not, the environment is too noisy for the row below to mean anything.
    let mut null_a = Vec::with_capacity(REPS);
    let mut null_b = Vec::with_capacity(REPS);
    let mut timings = Vec::with_capacity(REPS);
    let mut fused_timings = Vec::with_capacity(REPS);
    for sample in 0..REPS {
        if sample.is_multiple_of(2) {
            null_a.push(frankentorch_step(&base).0);
            null_b.push(frankentorch_step(&base).0);
            timings.push(frankentorch_step(&base).0);
            fused_timings.push(frankentorch_fused_step(&base).0);
        } else {
            null_b.push(frankentorch_step(&base).0);
            null_a.push(frankentorch_step(&base).0);
            fused_timings.push(frankentorch_fused_step(&base).0);
            timings.push(frankentorch_step(&base).0);
        }
    }
    let (null_ratio, null_low, null_high) = median_ratio_ci(&null_a, &null_b);
    let null_pass = null_low <= 1.0 && null_high >= 1.0;

    let frankentorch_ms = median(timings.clone());
    let allocator = if cfg!(feature = "fair-alloc") {
        "mimalloc (--features fair-alloc)"
    } else {
        "system (glibc malloc) — INFLATES the FrankenTorch arm, re-run with --features fair-alloc"
    };

    println!("executing_elf_sha256={}", executable_sha256());
    println!("allocator={allocator}");
    println!(
        "workload=avg_pool1d_grad_sum_loss [{N},{C},{L}] f64 kernel={KERNEL} stride={STRIDE} reps={REPS}"
    );
    println!(
        "a_a_median_ratio={null_ratio:.4} ci95=[{null_low:.4},{null_high:.4}] gate={}",
        if null_pass { "PASS" } else { "FAIL" }
    );

    // The ceiling of the "auto-route sum(avg_pool1d(x)) to the fused path" lever.
    // No routing change can beat calling the fused API directly, so if this ratio
    // does not clear 1.0 with its CI, the lever is not worth building.
    let fused_ms = median(fused_timings.clone());
    let (fused_ratio, fused_low, fused_high) = median_ratio_ci(&timings, &fused_timings);
    let lever_decision = if null_pass && fused_low > 1.0 {
        "LEVER HAS HEADROOM"
    } else {
        "LEVER REJECTED — fused is not measurably faster than the compose"
    };
    println!(
        "compose_ms={frankentorch_ms:.4} fused_ms={fused_ms:.4} compose_over_fused={fused_ratio:.4} ci95=[{fused_low:.4},{fused_high:.4}] {lever_decision}"
    );

    println!("op            FT(ms)    PT(ms)   verdict");
    // Anchor: a landed win reading its known value, so a bad measurement window is
    // visible rather than silently inflating the row under test.
    let mut anchor_best = f64::INFINITY;
    for _ in 0..7 {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let matrix = session
            .tensor_variable(anchor_values.clone(), vec![ANCHOR, ANCHOR], false)
            .expect("anchor leaf");
        let started = Instant::now();
        let _ = session.tensor_cat(&[matrix, matrix], 1);
        let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
        if elapsed < anchor_best {
            anchor_best = elapsed;
        }
    }
    for (name, ft) in [("cat_anchor", anchor_best), ("avg_pool1d", frankentorch_ms)] {
        if let Some(pt_ms) = pt_value(name) {
            let ratio = pt_ms / ft;
            let verdict = if ratio >= 1.0 {
                format!("FT {ratio:.2}x FASTER")
            } else {
                format!("FT {:.2}x SLOWER", 1.0 / ratio)
            };
            println!("  {name:<12} {ft:8.3} {pt_ms:8.3}   {verdict}");
        }
    }
    if !null_pass {
        println!(
            "  A/A NULL GATE FAILED — this window is too noisy; the avg_pool1d row above is not decidable."
        );
    }
    Ok(())
}
