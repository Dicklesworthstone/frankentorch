//! f32 group_norm (no-grad) FT vs PyTorch. cat_anchor = landed-win sanity. Build input outside timer.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example groupnorm_h2h

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use ft_kernel_cpu::group_norm_forward_f32;

const N: usize = 16;
const CH: usize = 256;
const H: usize = 64;
const W: usize = 64;
const G: usize = 32;
const R: usize = 4000;
const PARITY_INDICES: [usize; 8] = [0, 1, 255, 256, 4095, 1_048_575, 8_388_607, 16_777_215];
const REPS: usize = 31;
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
    let mut state = 0x6d8c_25a3_1f90_b7e2_u64;
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

fn elapsed_ms<F: FnOnce() -> Vec<f32>>(operation: F) -> f64 {
    let started = Instant::now();
    std::hint::black_box(operation());
    started.elapsed().as_secs_f64() * 1_000.0
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

fn scalar_group_norm_forward_f32(
    x: &[f32],
    batch: usize,
    num_groups: usize,
    cpg: usize,
    spatial: usize,
    eps: f32,
) -> Vec<f32> {
    let group_numel = cpg * spatial;
    let inv_m = 1.0 / group_numel as f32;
    let mut out = vec![0.0; x.len()];
    for group in 0..batch * num_groups {
        let base = group * group_numel;
        let input = &x[base..base + group_numel];
        let sum = input.iter().copied().sum::<f32>();
        let mean = sum * inv_m;
        let variance = input
            .iter()
            .map(|&value| {
                let delta = value - mean;
                delta * delta
            })
            .sum::<f32>();
        let rstd = 1.0 / (variance * inv_m + eps).sqrt();
        for (output, &value) in out[base..base + group_numel].iter_mut().zip(input) {
            *output = (value - mean) * rstd;
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let numel = N * CH * H * W;
    let x: Vec<f32> = (0..numel).map(|i| (i % 9973) as f32 - 4986.0).collect();
    let mat: Vec<f64> = (0..R * R).map(|i| (i % 17) as f64).collect();

    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
import torch.nn.functional as Fn
torch.set_num_threads(8)
N,CH,H,W,G,R=16,256,64,64,32,4000
x=((torch.arange(N*CH*H*W,dtype=torch.int64)%9973).float()-4986.0).reshape(N,CH,H,W)
m=((torch.arange(R*R,dtype=torch.int64)%17).double()).reshape(R,R)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([m,m],1)),("group_norm",lambda:Fn.group_norm(x,G))]:
    print("PT %s %.4f"%(name,t(fn)))
y=Fn.group_norm(x,G)
indices=[0,1,255,256,4095,1048575,8388607,16777215]
print("PT group_norm_probe",*("%.9g"%y.flatten()[i].item() for i in indices))
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
    let report = |name: &str, ftv: f64| {
        if let Some(p) = pt.lines().find_map(|l| {
            let mut it = l.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == name {
                it.next()?.parse::<f64>().ok()
            } else {
                None
            }
        }) {
            let r = p / ftv;
            let tag = if r >= 1.0 {
                format!("FT {r:.2}x FASTER")
            } else {
                format!("FT {:.2}x SLOWER", 1.0 / r)
            };
            println!("  {name:<12} {ftv:8.3} {p:8.3}   {tag}");
        }
    };
    let mut parity_session = FrankenTorchSession::new(ExecutionMode::Strict);
    let parity_input = parity_session
        .tensor_variable_f32(x.clone(), vec![N, CH, H, W], false)
        .unwrap();
    let parity_output = parity_session
        .functional_group_norm(parity_input, G, None, None, 1e-5)
        .unwrap();
    let parity_values = parity_session.tensor_values_f32(parity_output).unwrap();
    let python_probe = pt
        .lines()
        .find_map(|line| line.strip_prefix("PT group_norm_probe "))
        .expect("PyTorch GroupNorm parity probe missing")
        .split_whitespace()
        .map(str::parse::<f32>)
        .collect::<Result<Vec<_>, _>>()?;
    assert_eq!(python_probe.len(), PARITY_INDICES.len());
    for (&index, &expected) in PARITY_INDICES.iter().zip(&python_probe) {
        let actual = parity_values[index];
        let tolerance = 2e-5_f32 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "f32 GroupNorm parity at {index}: got {actual}, PyTorch {expected}, tolerance {tolerance}"
        );
    }
    println!(
        "  group_norm f32 parity: {} PyTorch probes within tolerance",
        PARITY_INDICES.len()
    );
    let candidate = || group_norm_forward_f32(&x, None, None, N, G, CH / G, H * W, 1e-5);
    let incumbent = || scalar_group_norm_forward_f32(&x, N, G, CH / G, H * W, 1e-5);
    assert_eq!(
        candidate()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        incumbent()
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>(),
        "SIMD candidate must preserve the scalar kernel exactly"
    );
    for _ in 0..4 {
        std::hint::black_box(incumbent());
        std::hint::black_box(candidate());
    }
    let mut null_a = Vec::with_capacity(REPS);
    let mut null_b = Vec::with_capacity(REPS);
    let mut old = Vec::with_capacity(REPS);
    let mut new = Vec::with_capacity(REPS);
    for sample in 0..REPS {
        if sample.is_multiple_of(2) {
            null_a.push(elapsed_ms(candidate));
            null_b.push(elapsed_ms(candidate));
            old.push(elapsed_ms(incumbent));
            new.push(elapsed_ms(candidate));
        } else {
            null_b.push(elapsed_ms(candidate));
            null_a.push(elapsed_ms(candidate));
            new.push(elapsed_ms(candidate));
            old.push(elapsed_ms(incumbent));
        }
    }
    let (null_ratio, null_low, null_high) = median_ratio_ci(&null_a, &null_b);
    let (speedup, speedup_low, speedup_high) = median_ratio_ci(&old, &new);
    let null_pass = null_low <= 1.0 && null_high >= 1.0;
    let decision = if null_pass && speedup_low > 1.0 {
        "KEEP"
    } else {
        "REJECT"
    };
    let kernel_report = format!(
        "executing_elf_sha256={}\nworkload=group_norm_f32_no_affine [16,256,64,64] groups=32 reps={REPS}\na_a_median_ratio={null_ratio:.4} ci95=[{null_low:.4},{null_high:.4}] gate={}\nscalar_ms={:.4} simd_ms={:.4} scalar_over_simd={speedup:.4} ci95=[{speedup_low:.4},{speedup_high:.4}] decision={decision}\n",
        executable_sha256(),
        if null_pass { "PASS" } else { "FAIL" },
        median(old.clone()),
        median(new.clone()),
    );
    if let Ok(target_dir) = std::env::var("CARGO_TARGET_DIR") {
        let _ = std::fs::write(
            format!("{target_dir}/groupnorm_h2h_f32_ab.txt"),
            &kernel_report,
        );
    }
    print!("{kernel_report}");
    println!(
        "  f32 kernel A/A={null_ratio:.4} ci95=[{null_low:.4},{null_high:.4}] {} | scalar/SIMD={speedup:.4} ci95=[{speedup_low:.4},{speedup_high:.4}] {}",
        if null_pass { "PASS" } else { "FAIL" },
        decision,
    );
    println!("op            FT(ms)    PT(ms)   verdict");
    // cat anchor
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xm = s.tensor_variable(mat.clone(), vec![R, R], false).unwrap();
        let t = Instant::now();
        let _ = s.tensor_cat(&[xm, xm], 1);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < b {
            b = e;
        }
    }
    report("cat_anchor", b);
    let mut b = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xn = s
            .tensor_variable_f32(x.clone(), vec![N, CH, H, W], false)
            .unwrap();
        let t = Instant::now();
        let _ = s.functional_group_norm(xn, G, None, None, 1e-5);
        let e = t.elapsed().as_secs_f64() * 1e3;
        if e < b {
            b = e;
        }
    }
    report("group_norm", b);
    Ok(())
}
