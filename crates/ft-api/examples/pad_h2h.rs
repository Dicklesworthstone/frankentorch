//! Padding op scan FT vs PyTorch ([4000,4000] f64 no-grad, pad 16 each side) — hunting the
//! per-element division-unravel anti-pattern in constant/reflect/replicate pad. `cat` is a
//! landed-win ANCHOR for worker-health (discard the run if it regresses far from ~3-6x).
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example pad_h2h

use std::io::{BufRead, BufReader, Write};
use std::process::{ChildStdin, ChildStdout, Command, Stdio};
use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_api::harness_interleave::{
    BALANCED_SQUARE, QUIT_REQUEST, READY_MARKER, SAMPLE_LOOP_PY, interpreter_args,
    parse_sample_line, sample_request,
};
use ft_core::ExecutionMode;

const R: usize = 4000;
const C: usize = 4000;
const P: usize = 16;
const GRAD_REPS: usize = 12;
const GRAD_NULL_MIN: f64 = 0.97;
const GRAD_NULL_MAX: f64 = 1.03;

type UnaryOp = fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId);

fn time_ft<F: Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId)>(data: &[f64], f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable(data.to_vec(), vec![R, C], false).unwrap();
        let t = Instant::now();
        f(&mut s, x);
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
        }
    }
    best
}

fn time_ft_f32_constant_pad() -> f64 {
    let data: Vec<f32> = (0..R * C).map(|i| ((i % 17) as f32) - 8.0).collect();
    let mut best = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(data.clone(), vec![R, C], false)
            .unwrap();
        let t = Instant::now();
        let _ = s.tensor_pad(x, &[P, P, P, P], 0.0);
        let el = t.elapsed().as_secs_f64() * 1e3;
        if el < best {
            best = el;
        }
    }
    best
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    if values.len().is_multiple_of(2) {
        (values[middle - 1] + values[middle]) * 0.5
    } else {
        values[middle]
    }
}

fn paired_slot_median(mut values: [f64; 2]) -> f64 {
    values.sort_by(f64::total_cmp);
    (values[0] + values[1]) * 0.5
}

fn grad_reps() -> usize {
    std::env::var("FT_GRAD_PAD_REPS")
        .ok()
        .and_then(|raw| raw.trim().parse().ok())
        .filter(|reps: &usize| *reps >= 8)
        .unwrap_or(GRAD_REPS)
}

/// Builds the leaf outside the clock, then prices only the graph-producing pad forward.
///
/// `wb7vt` specifically removed the no-grad guard from the block-copy path, so this must keep
/// `requires_grad=true` even though the backward crop is intentionally outside this forward gate.
fn timed_grad_constant_pad(data: &[f32], r: usize, c: usize, p: usize) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let input = session
        .tensor_variable_f32(data.to_vec(), vec![r, c], true)
        .expect("grad-pad input");
    let started = Instant::now();
    let output = session
        .tensor_pad(input, &[p, p, p, p], 0.0)
        .expect("grad-pad forward");
    let milliseconds = started.elapsed().as_secs_f64() * 1_000.0;
    assert!(
        session
            .tensor_requires_grad(output)
            .expect("grad-pad output grad flag"),
        "the measured pad must retain its autograd node"
    );
    let checksum = session
        .tensor_values_f32(output)
        .expect("grad-pad output values")
        .iter()
        .map(|value| f64::from(*value))
        .sum();
    (milliseconds, checksum)
}

/// Builds the graph outside the clock, then prices only the Pad crop in backward.
fn timed_grad_constant_pad_backward(data: &[f32], r: usize, c: usize, p: usize) -> (f64, f64) {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let input = session
        .tensor_variable_f32(data.to_vec(), vec![r, c], true)
        .expect("grad-pad input");
    let output = session
        .tensor_pad(input, &[p, p, p, p], 0.0)
        .expect("grad-pad forward");
    let loss = session.tensor_sum(output).expect("grad-pad loss");
    let started = Instant::now();
    let report = session.tensor_backward(loss).expect("grad-pad backward");
    let milliseconds = started.elapsed().as_secs_f64() * 1_000.0;
    let checksum = report
        .gradient(input)
        .expect("grad-pad input gradient")
        .iter()
        .sum();
    (milliseconds, checksum)
}

fn incumbent_grad_pad_sample(
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
            return Err("PyTorch co-process closed before its grad-pad sample".into());
        }
        if let Some(sample) = parse_sample_line(&line) {
            assert_eq!(sample.lane, lane);
            return Ok((sample.milliseconds, sample.gradient_checksum));
        }
    }
}

fn run_grad_pad_h2h() -> Result<(), Box<dyn std::error::Error>> {
    let data: Vec<f32> = (0..R * C).map(|i| ((i % 17) as f32) - 8.0).collect();
    let backward_only = std::env::var_os("FT_GRAD_PAD_BACKWARD_H2H").is_some();
    let lane = if backward_only {
        "constant_pad_f32_grad_backward"
    } else {
        "constant_pad_f32_grad"
    };
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py_setup = r#"
import time, torch
import torch.nn.functional as F
print('PT_TORCH_VERSION %s' % torch.__version__, flush=True)
torch.set_num_threads(8)
R,C,P=4000,4000,16
idx=torch.arange(R*C,dtype=torch.int64)
pad_input=((idx%17).float()-8.0).reshape(R,C).requires_grad_(True)
def run(base, fn):
    s=time.perf_counter()
    out=fn(base)
    elapsed=(time.perf_counter()-s)*1e3
    assert out.requires_grad
    return elapsed, out.detach().sum().item()
def run_backward(base, fn):
    inp=base.detach().requires_grad_(True)
    out=fn(inp)
    loss=out.sum()
    s=time.perf_counter()
    loss.backward()
    elapsed=(time.perf_counter()-s)*1e3
    return elapsed, inp.grad.detach().sum().item()
print('PT_TIMED_STEPS forward_with_grad_graph', flush=True)
print('PT_TIMED_STEPS backward_after_pad_graph', flush=True)
LANES = {
    "constant_pad_f32_grad": (pad_input, lambda x: F.pad(x,(P,P,P,P),mode='constant',value=0.0)),
}
"#;
    let backward_sample_loop = r#"
import sys, os
for _ in range(int(os.environ.get('FT_H2H_WARMUP', '32'))):
    run_backward(pad_input, lambda x: F.pad(x,(P,P,P,P),mode='constant',value=0.0))
print('PT_READY', flush=True)
for _line in sys.stdin:
    _line = _line.strip()
    if _line == 'QUIT':
        break
    if _line == 'SAMPLE constant_pad_f32_grad_backward':
        _ms, _g = run_backward(pad_input, lambda x: F.pad(x,(P,P,P,P),mode='constant',value=0.0))
        print('PT_SAMPLE constant_pad_f32_grad_backward %.6f %.12g' % (_ms, _g), flush=True)
"#;
    let py = if backward_only {
        format!("{py_setup}{backward_sample_loop}")
    } else {
        format!("{py_setup}{SAMPLE_LOOP_PY}")
    };
    let mut child = Command::new(&python)
        .args(interpreter_args(&py))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let mut stdin = child
        .stdin
        .take()
        .ok_or_else(|| std::io::Error::other("no PyTorch stdin"))?;
    let mut reader = BufReader::new(
        child
            .stdout
            .take()
            .ok_or_else(|| std::io::Error::other("no PyTorch stdout"))?,
    );
    let mut preamble = String::new();
    loop {
        let mut line = String::new();
        if reader.read_line(&mut line)? == 0 {
            return Err(
                format!("PyTorch arm ({python}) exited before {READY_MARKER}: {preamble}").into(),
            );
        }
        if line.trim() == READY_MARKER {
            break;
        }
        preamble.push_str(&line);
    }
    let torch_version = ft_api::harness_provenance::require_reported_version(&preamble)?;
    let timed_step = if backward_only {
        "PT_TIMED_STEPS backward_after_pad_graph"
    } else {
        "PT_TIMED_STEPS forward_with_grad_graph"
    };
    assert!(
        preamble.contains(timed_step),
        "the PyTorch arm must report its timed step"
    );
    println!(
        "executing_elf_sha256={}",
        ft_api::harness_provenance::executing_elf_sha256()
    );
    println!(
        "{}",
        ft_api::harness_provenance::incumbent_provenance_block(torch_version, 8)
    );
    println!(
        "measurement=CONSTANT PAD f32 {} WITH requires_grad=true; graph construction/checksum outside timer; balanced-square interleaved same invocation",
        if backward_only { "BACKWARD" } else { "FORWARD" }
    );

    for _ in 0..std::env::var("FT_H2H_WARMUP")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(4)
    {
        std::hint::black_box(if backward_only {
            timed_grad_constant_pad_backward(&data, R, C, P)
        } else {
            timed_grad_constant_pad(&data, R, C, P)
        });
    }
    let reps = grad_reps();
    let mut ft_times = Vec::with_capacity(reps);
    let mut pt_times = Vec::with_capacity(reps);
    let mut ft_first_half = Vec::with_capacity(reps);
    let mut ft_second_half = Vec::with_capacity(reps);
    let mut pt_first_half = Vec::with_capacity(reps);
    let mut pt_second_half = Vec::with_capacity(reps);
    let mut ft_checksum = 0.0;
    let mut pt_checksum = 0.0;
    for _ in 0..reps {
        let mut ft_slots = Vec::with_capacity(4);
        let mut pt_slots = Vec::with_capacity(4);
        for incumbent_slot in BALANCED_SQUARE {
            if incumbent_slot {
                let (milliseconds, checksum) =
                    incumbent_grad_pad_sample(&mut stdin, &mut reader, lane)?;
                pt_slots.push(milliseconds);
                pt_checksum = checksum;
            } else {
                let (milliseconds, checksum) = if backward_only {
                    timed_grad_constant_pad_backward(&data, R, C, P)
                } else {
                    timed_grad_constant_pad(&data, R, C, P)
                };
                ft_slots.push(milliseconds);
                ft_checksum = checksum;
            }
        }
        ft_first_half.push(paired_slot_median([ft_slots[0], ft_slots[1]]));
        ft_second_half.push(paired_slot_median([ft_slots[2], ft_slots[3]]));
        pt_first_half.push(paired_slot_median([pt_slots[0], pt_slots[1]]));
        pt_second_half.push(paired_slot_median([pt_slots[2], pt_slots[3]]));
        ft_times.push(median(ft_slots));
        pt_times.push(median(pt_slots));
    }
    writeln!(stdin, "{QUIT_REQUEST}")?;
    stdin.flush()?;
    drop(stdin);
    child.wait()?;

    let ratio = median(pt_times.clone()) / median(ft_times.clone());
    let pt_null = median(pt_first_half) / median(pt_second_half);
    let ft_null = median(ft_first_half) / median(ft_second_half);
    let parity = (ft_checksum - pt_checksum).abs() <= 1e-5 * pt_checksum.abs().max(1.0);
    let nulls_pass = (GRAD_NULL_MIN..=GRAD_NULL_MAX).contains(&pt_null)
        && (GRAD_NULL_MIN..=GRAD_NULL_MAX).contains(&ft_null);
    println!(
        "{lane} {R}x{C} pad={P}: FT {:.3} ms PT {:.3} ms = FT {:.3}x {} | PT A/A {pt_null:.3} {} FT A/A {ft_null:.3} {} parity {}",
        median(ft_times),
        median(pt_times),
        if ratio >= 1.0 { ratio } else { 1.0 / ratio },
        if ratio >= 1.0 { "FASTER" } else { "SLOWER" },
        if (GRAD_NULL_MIN..=GRAD_NULL_MAX).contains(&pt_null) {
            "PASS"
        } else {
            "FAIL"
        },
        if (GRAD_NULL_MIN..=GRAD_NULL_MAX).contains(&ft_null) {
            "PASS"
        } else {
            "FAIL"
        },
        if parity { "MATCH" } else { "MISMATCH" },
    );
    if !(nulls_pass && parity) {
        println!(
            "NOT QUOTABLE: require both duplicate-arm A/A nulls inside {GRAD_NULL_MIN:.2}..{GRAD_NULL_MAX:.2} and checksum parity"
        );
    }
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if std::env::var_os("FT_GRAD_PAD_H2H").is_some()
        || std::env::var_os("FT_GRAD_PAD_BACKWARD_H2H").is_some()
    {
        return run_grad_pad_h2h();
    }
    let data: Vec<f64> = (0..R * C).map(|i| ((i % 17) as f64) - 8.0).collect();
    let ops: Vec<(&str, UnaryOp)> = vec![
        ("cat_anchor", |s, x| {
            let _ = s.tensor_cat(&[x, x], 1);
        }),
        ("constant_pad", |s, x| {
            let _ = s.tensor_pad(x, &[P, P, P, P], 0.0);
        }),
        ("reflect_pad", |s, x| {
            let _ = s.tensor_pad_mode(x, &[P, P, P, P], "reflect", 0.0);
        }),
        ("replicate_pad", |s, x| {
            let _ = s.tensor_pad_mode(x, &[P, P, P, P], "replicate", 0.0);
        }),
    ];
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    let py = r#"
import time, torch
import torch.nn.functional as F
torch.set_num_threads(8)
R,C,P=4000,4000,16
idx=torch.arange(R*C,dtype=torch.int64)
x=((idx%17).double()-8.0).reshape(R,C)
xf=((idx%17).float()-8.0).reshape(R,C)
def t(fn,n=7):
    for _ in range(2):
        try: fn()
        except Exception as e: return float('nan')
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
for name,fn in [("cat_anchor",lambda:torch.cat([x,x],1)),
                ("constant_pad",lambda:F.pad(x,(P,P,P,P),mode='constant',value=0.0)),
                ("constant_pad_f32",lambda:F.pad(xf,(P,P,P,P),mode='constant',value=0.0)),
                ("reflect_pad",lambda:F.pad(x,(P,P,P,P),mode='reflect')),
                ("replicate_pad",lambda:F.pad(x,(P,P,P,P),mode='replicate'))]:
    print("PT %s %.4f"%(name,t(fn)))
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
    println!("op            FT(ms)    PT(ms)   ratio(PT/FT, <1=FT slower)");
    let lookup = |name: &str| -> Option<f64> {
        pt.lines().find_map(|l| {
            let mut it = l.strip_prefix("PT ")?.split_whitespace();
            if it.next()? == name {
                it.next()?.parse::<f64>().ok()
            } else {
                None
            }
        })
    };
    let report = |name: &str, ftv: f64, p: Option<f64>| {
        if let Some(p) = p {
            let r = p / ftv;
            let tag = if r >= 1.0 {
                format!("FT {r:.2}x FASTER")
            } else {
                format!("FT {:.2}x SLOWER", 1.0 / r)
            };
            println!("  {name:<16} {ftv:8.3} {p:8.3}   {tag}");
        }
    };
    for (name, f) in &ops {
        let ftv = time_ft(&data, *f);
        report(name, ftv, lookup(name));
    }
    report(
        "constant_pad_f32",
        time_ft_f32_constant_pad(),
        lookup("constant_pad_f32"),
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{median, paired_slot_median, timed_grad_constant_pad};

    #[test]
    fn grad_pad_sample_keeps_the_pad_node_and_prices_only_forward() {
        let (milliseconds, checksum) = timed_grad_constant_pad(&[1.0, 2.0, 3.0, 4.0], 2, 2, 1);
        assert!(milliseconds.is_finite() && milliseconds >= 0.0);
        assert_eq!(checksum.to_bits(), 10.0_f64.to_bits());
    }

    #[test]
    fn balanced_slot_medians_are_order_independent() {
        assert_eq!(median(vec![9.0, 1.0, 3.0, 7.0]), 5.0);
        assert_eq!(paired_slot_median([9.0, 1.0]), 5.0);
    }
}
