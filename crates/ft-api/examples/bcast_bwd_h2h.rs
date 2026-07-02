use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn fingerprint(g: &[f64]) -> u64 {
    let mut h: u64 = 1469598103934665603;
    for &x in g {
        h = h.wrapping_mul(1099511628211) ^ x.to_bits();
    }
    h
}

// Broadcast-backward (bias-gradient): the reduction half of broadcasting.
fn main() {
    let tag = std::env::var("FT_TAG").unwrap_or_else(|_| "FT".into());
    let (rows, cols) = (4096usize, 4096usize);
    let n = rows * cols;
    let da: Vec<f64> = (0..n).map(|i| 0.5 + ((i % 971) as f64) * 0.01).collect();
    // b shapes: row-vector [1,cols] and col-vector [rows,1]
    for (name, bshape, bn) in [("bias[1,C]", vec![1usize, cols], cols), ("bias[R,1]", vec![rows, 1usize], rows)] {
        let db: Vec<f64> = (0..bn).map(|i| 0.1 + ((i % 617) as f64) * 0.002).collect();
        // varying, non-grad weight so the gradient reaching b's broadcast-reduction is
        // NON-constant -> the fingerprint actually checks the summation order.
        let dw: Vec<f64> = (0..n).map(|i| 0.3 + ((i % 733) as f64) * 0.001).collect();
        let mut best = f64::INFINITY;
        let mut best_fwd = f64::INFINITY;
        let mut best_bwd = f64::INFINITY;
        let mut fpb = 0u64;
        let mut gb0 = 0u64;
        let mut gbl = 0u64;
        for _ in 0..7 {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let _ = &da;
            let b = s.tensor_variable(db.clone(), bshape.clone(), true).unwrap();
            let w = s.tensor_variable(dw.clone(), vec![rows, cols], false).unwrap();
            let t0 = Instant::now();
            // Pure Expand backward isolation: broadcast b -> [rows,cols], weight, reduce.
            let c = s.tensor_expand(b, vec![rows, cols]).unwrap();
            let d = s.tensor_mul(c, w).unwrap();
            let loss = s.tensor_sum(d).unwrap();
            let t1 = Instant::now();
            s.tensor_backward(loss).unwrap();
            let t2 = Instant::now();
            let ms = (t2 - t0).as_secs_f64() * 1e3;
            if ms < best {
                best = ms;
                best_fwd = (t1 - t0).as_secs_f64() * 1e3;
                best_bwd = (t2 - t1).as_secs_f64() * 1e3;
                if let Ok(Some(g)) = s.tensor_grad(b) {
                    fpb = fingerprint(&g);
                    gb0 = g[0].to_bits();
                    gbl = g[g.len() - 1].to_bits();
                }
            }
            std::hint::black_box(&s);
        }
        println!("[{tag}] add+{name} f64 [4096,4096]: total {best:.2} ms (fwd {best_fwd:.2} + bwd {best_bwd:.2}) | grad_b_fp=0x{fpb:016x} gb[0]=0x{gb0:016x} gb[-1]=0x{gbl:016x}");
    }
}
