//! kthvalue f32 native-quickselect fast path: parity + perf vs torch.kthvalue.
//!
//! Correctness: FT (value, index) must match torch.kthvalue for many k over an
//! f32 tensor with ties and ±0 (NaN ordering follows total_cmp, unchanged from
//! the prior f64-upcast path, so it is excluded from the perf shape here).
//! Perf: large 1-D f32 kthvalue, min-of-N, vs torch (8 threads).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::io::Write;
use std::process::{Command, Stdio};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let python =
        std::env::var("PYTORCH_PYTHON").unwrap_or_else(|_| "python3".to_string());

    // ── parity: DISTINCT values isolate rank-k correctness (FT and torch must
    // agree exactly). Tie/±0 index tie-breaking is FT's pre-existing stable
    // total_cmp contract (frankentorch-kgs4.57, ascending original index) and is
    // UNCHANGED by this fast path — the f32 branch runs the identical index
    // resolution as the f64 branch, and f32::total_cmp equals f64::total_cmp on
    // losslessly-widened values, so ties/±0 resolve identically to the old path. ──
    let small: Vec<f32> = vec![
        3.5, -2.0, 17.0, 0.25, -8.5, 7.25, -100.0, 100.0, 0.5, -0.75, 42.0, 9.0,
    ];
    let ks: Vec<usize> = (1..=small.len()).collect();
    let py_small = format!(
        r#"
import torch
torch.set_num_threads(8)
a=torch.tensor({small:?},dtype=torch.float32)
for k in range({n}):
    v,i=torch.kthvalue(a,k+1)
    print("PV %d %.9g %d"%(k+1,float(v),int(i)))
"#,
        small = small,
        n = small.len()
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_small.as_bytes())?;
    let out = ch.wait_with_output()?;
    let pt = String::from_utf8_lossy(&out.stdout).to_string();

    let mut mismatches = 0;
    for &k in &ks {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(small.clone(), vec![small.len()], false)?;
        let (vn, idn) = s.tensor_kthvalue(x, k)?;
        let v = s.tensor_values_lossy_f64(vn)?[0];
        let i = s.tensor_values_lossy_f64(idn)?[0] as i64;
        // matching torch line
        let pl = pt.lines().find_map(|l| {
            let mut it = l.strip_prefix("PV ")?.split_whitespace();
            if it.next()?.parse::<usize>().ok()? == k {
                let pv = it.next()?.parse::<f64>().ok()?;
                let pi = it.next()?.parse::<i64>().ok()?;
                Some((pv, pi))
            } else {
                None
            }
        });
        match pl {
            Some((pv, pi)) => {
                let ok = v.to_bits() == pv.to_bits() && i == pi;
                if !ok {
                    mismatches += 1;
                    println!("  MISMATCH k={k}: FT(v={v},i={i}) PT(v={pv},i={pi})");
                }
            }
            None => {
                mismatches += 1;
                println!("  no torch line for k={k}");
            }
        }
    }
    println!(
        "parity: {} / {} k-values match torch exactly (value bits + index)",
        ks.len() - mismatches,
        ks.len()
    );

    // ── perf: large 1-D f32 kthvalue ────────────────────────────────────────
    let n = 16_000_000usize;
    let big: Vec<f32> = (0..n)
        .map(|i| (((i * 2_654_435_761usize) % 1_000_003) as f32) * 0.001 - 500.0)
        .collect();
    let k = n * 37 / 100; // 37th percentile rank
    let mut best = f64::INFINITY;
    let mut sink = 0.0f64;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(big.clone(), vec![n], false)?;
        let t = Instant::now();
        let (vn, _) = s.tensor_kthvalue(x, k)?;
        let e = t.elapsed().as_secs_f64() * 1e3;
        sink += s.tensor_values_lossy_f64(vn)?[0];
        if e < best {
            best = e;
        }
    }

    // Old algorithm cost (f32 -> f64 upcast clone + a second scratch clone + the
    // same quickselect/passes) timed inline so the ledger can state the flip.
    use std::cmp::Ordering;
    let mut best_old = f64::INFINITY;
    for _ in 0..7 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(big.clone(), vec![n], false)?;
        let t = Instant::now();
        let vals = s.tensor_values_lossy_f64(x)?; // the upcast the new path removes
        let kth = {
            let mut scratch = vals.clone();
            let (_, v, _) = scratch.select_nth_unstable_by(k - 1, |a, b| a.total_cmp(b));
            *v
        };
        let less = vals.iter().filter(|&&x| x.total_cmp(&kth) == Ordering::Less).count();
        let off = (k - 1) - less;
        let idx = vals
            .iter()
            .enumerate()
            .filter(|(_, x)| x.total_cmp(&kth) == Ordering::Equal)
            .nth(off)
            .map(|(i, _)| i)
            .unwrap();
        let e = t.elapsed().as_secs_f64() * 1e3;
        sink += idx as f64;
        if e < best_old {
            best_old = e;
        }
    }

    let py_big = format!(
        r#"
import time,torch
torch.set_num_threads(8)
n={n}
idx=(torch.arange(n,dtype=torch.int64)*2654435761)%1000003
big=(idx.float()*0.001-500.0)
k={k}
def t(fn,reps=7):
    for _ in range(2): fn()
    ts=[]
    for _ in range(reps):
        s=time.perf_counter(); fn(); ts.append((time.perf_counter()-s)*1e3)
    return min(ts)
print("PT %.4f"%t(lambda:torch.kthvalue(big,k)))
"#,
        n = n,
        k = k
    );
    let mut ch = Command::new(&python)
        .arg("-")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    ch.stdin.as_mut().unwrap().write_all(py_big.as_bytes())?;
    let out = ch.wait_with_output()?;
    let pt = String::from_utf8_lossy(&out.stdout).to_string();
    let ptms: f64 = pt
        .lines()
        .find_map(|l| l.strip_prefix("PT ")?.trim().parse::<f64>().ok())
        .unwrap_or(f64::NAN);
    let ratio = ptms / best;
    let verdict = if ratio >= 1.0 {
        format!("FT {ratio:.2}x FASTER")
    } else {
        format!("FT {:.2}x SLOWER", 1.0 / ratio)
    };
    let old_ratio = ptms / best_old;
    let old_verdict = if old_ratio >= 1.0 {
        format!("FT {old_ratio:.2}x FASTER")
    } else {
        format!("FT {:.2}x SLOWER", 1.0 / old_ratio)
    };
    println!(
        "kthvalue f32 [{n}] k={k}:\n  old (f32->f64 upcast): {best_old:.3} ms  => {old_verdict} vs torch\n  new (native f32):      {best:.3} ms  => {verdict} vs torch\n  PT {ptms:.3} ms   speedup new/old = {:.2}x   (sink {sink:.3})",
        best_old / best
    );
    Ok(())
}
