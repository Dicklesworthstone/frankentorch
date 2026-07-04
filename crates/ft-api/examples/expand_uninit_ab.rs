//! Same-process, same-worker A/B for the expand/broadcast_to materialization lever:
//! OLD = `vec![values[0]; n]` (serial first-touch) + par_chunks_mut fill;
//! NEW = `ft_kernel_cpu::expand_row_structured` (uninit alloc, fill does first-touch).
//! Both paths + a torch broadcast_to reference, all in one process. min-of-9.
//! Run: PYTORCH_PYTHON=/path/to/python cargo run --release -p ft-api --example expand_uninit_ab

use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

// Local copy of the OLD ft-autograd path (baseline for the pure-lever A/B).
fn old_expand<T: Copy + Send + Sync>(
    values: &[T],
    output_numel: usize,
    target_shape: &[usize],
    input_strides: &[usize],
) -> Vec<T> {
    let nd = target_shape.len();
    let inner = target_shape[nd - 1];
    let inner_stride = input_strides[nd - 1];
    let mut outer_strides = vec![1usize; nd.saturating_sub(1)];
    for d in (0..nd.saturating_sub(1)).rev() {
        if d + 1 < nd - 1 {
            outer_strides[d] = outer_strides[d + 1] * target_shape[d + 1];
        }
    }
    let row_base = |r: usize| -> usize {
        let mut idx = 0usize;
        for d in 0..nd - 1 {
            let c = (r / outer_strides[d]) % target_shape[d];
            idx += c * input_strides[d];
        }
        idx
    };
    let mut output = vec![values[0]; output_numel];
    let fill_row = |r: usize, row: &mut [T]| {
        let base = row_base(r);
        if inner_stride == 0 {
            row.fill(values[base]);
        } else {
            row.copy_from_slice(&values[base..base + inner]);
        }
    };
    const PAR_MIN: usize = 1 << 16;
    if output_numel >= PAR_MIN {
        use rayon::prelude::*;
        output
            .par_chunks_mut(inner)
            .enumerate()
            .for_each(|(r, row)| fill_row(r, row));
    } else {
        for (r, row) in output.chunks_mut(inner).enumerate() {
            fill_row(r, row);
        }
    }
    output
}

// contiguous-input broadcast strides for `in_shape` -> `target_shape` (same rank).
fn bcast_strides(in_shape: &[usize], target_shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1usize; in_shape.len()];
    for d in (0..in_shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * in_shape[d + 1];
    }
    (0..in_shape.len())
        .map(|d| if in_shape[d] == target_shape[d] { s[d] } else { 0 })
        .collect()
}

fn bench<F: Fn() -> usize>(f: F) -> f64 {
    let mut best = f64::INFINITY;
    for _ in 0..9 {
        let t = Instant::now();
        let s = f();
        let el = t.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(s);
        if el < best {
            best = el;
        }
    }
    best
}

fn main() {
    // (label, in_shape, target_shape)
    let cases: Vec<(&str, Vec<usize>, Vec<usize>)> = vec![
        ("row[1,2048]->[2048,2048]", vec![1, 2048], vec![2048, 2048]),
        ("row[1,4096]->[4096,4096]", vec![1, 4096], vec![4096, 4096]),
        ("col[4096,1]->[4096,4096]", vec![4096, 1], vec![4096, 4096]),
        ("row[1,8192]->[8192,8192]", vec![1, 8192], vec![8192, 8192]),
    ];
    let python = std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());
    // torch reference times
    let py = r#"
import time, torch
torch.set_num_threads(8)
def tm(fn,n=9):
    for _ in range(3): fn()
    ts=[]
    for _ in range(n):
        s=time.perf_counter(); r=fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
def run(insh, tgt):
    x=torch.arange(1, 1+int(torch.tensor(insh).prod()), dtype=torch.float32).reshape(insh)
    return tm(lambda: torch.broadcast_to(x, tgt).contiguous())
for name,insh,tgt in [("row2048",[1,2048],[2048,2048]),("row4096",[1,4096],[4096,4096]),
                      ("col4096",[4096,1],[4096,4096]),("row8192",[1,8192],[8192,8192])]:
    print("PT %s %.4f"%(name, run(insh,tgt)))
"#;
    let pt = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .and_then(|mut c| {
            c.stdin.as_mut().unwrap().write_all(py.as_bytes())?;
            let o = c.wait_with_output()?;
            Ok(String::from_utf8_lossy(&o.stdout).to_string())
        })
        .unwrap_or_default();
    let ptval = |k: &str| -> f64 {
        pt.lines()
            .find_map(|l| {
                let mut it = l.strip_prefix("PT ")?.split_whitespace();
                if it.next()? == k {
                    it.next()?.parse::<f64>().ok()
                } else {
                    None
                }
            })
            .unwrap_or(f64::NAN)
    };
    let pt_keys = ["row2048", "row4096", "col4096", "row8192"];

    println!(
        "case                         OLD(ms)  NEW(ms)  NEW/OLD    torch(ms)  NEW vs torch   [f32, min-9]"
    );
    for (i, (label, insh, tgt)) in cases.iter().enumerate() {
        let numel: usize = tgt.iter().product();
        let in_numel: usize = insh.iter().product();
        let values: Vec<f32> = (0..in_numel).map(|k| (k % 997) as f32 + 0.5).collect();
        let strides = bcast_strides(insh, tgt);
        // correctness: NEW == OLD bit-for-bit
        let a = old_expand(&values, numel, tgt, &strides);
        let b = ft_kernel_cpu::expand_row_structured(&values, numel, tgt, &strides);
        let bitmatch = a == b;
        let old_ms = bench(|| old_expand(&values, numel, tgt, &strides).len());
        let new_ms = bench(|| ft_kernel_cpu::expand_row_structured(&values, numel, tgt, &strides).len());
        let ptms = ptval(pt_keys[i]);
        let ratio = old_ms / new_ms;
        let vs_torch = if ptms.is_finite() {
            if ptms >= new_ms {
                format!("{:.2}x FASTER", ptms / new_ms)
            } else {
                format!("{:.2}x SLOWER", new_ms / ptms)
            }
        } else {
            "n/a".to_string()
        };
        println!(
            "  {label:<28} {old_ms:7.3} {new_ms:7.3}   {ratio:5.2}x    {ptms:7.3}   {vs_torch:<14} bitmatch={bitmatch}"
        );
    }
}
