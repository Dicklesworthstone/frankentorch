//! Real-op A/B for apply_rotary_pos_emb (RoPE). OLD = replicate the compose (q*cos + rotate_half(q)*sin
//! via mul/mul/add, trailing [S,D] tile broadcast); NEW = s.apply_rotary_pos_emb (fused no-grad F64 path).
//! bitmatch verifies fused == compose. Run: cargo run --release -p ft-api --example rope_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

// Replica of the compose: out[f] = q[f]*cos[f%tile] + rot[f]*sin[f%tile], rot = cat(-x2, x1).
fn old_rope(q: &[f64], cos: &[f64], sin: &[f64], head_dim: usize, tile: usize) -> Vec<f64> {
    let half = head_dim / 2;
    // q_cos = q*cos (mul), q_rot = rotate_half(q), q_sin = q_rot*sin (mul), out = q_cos + q_sin (add)
    let q_cos: Vec<f64> = q
        .iter()
        .enumerate()
        .map(|(f, &v)| v * cos[f % tile])
        .collect();
    let mut q_rot = vec![0.0; q.len()];
    for (f, r) in q_rot.iter_mut().enumerate() {
        let i = f % head_dim;
        *r = if i < half { -q[f + half] } else { q[f - half] };
    }
    let q_sin: Vec<f64> = q_rot
        .iter()
        .enumerate()
        .map(|(f, &v)| v * sin[f % tile])
        .collect();
    q_cos
        .iter()
        .zip(q_sin.iter())
        .map(|(&a, &b)| a + b)
        .collect()
}

fn bench<F: FnMut() -> usize>(mut f: F) -> f64 {
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
    println!("apply_rotary_pos_emb (RoPE) f64, min-9:  OLD=compose(6 ops/qk)  NEW=fused-parallel");
    let cases = [
        ("B8 H8 S512 D64", 8usize, 8, 512, 64),
        ("B16 H16 S256 D64", 16, 16, 256, 64),
        ("B4 H32 S1024 D128", 4, 32, 1024, 128),
    ];
    for (label, b, h, s, d) in cases {
        let tile = s * d;
        let n = b * h * tile;
        let q: Vec<f64> = (0..n).map(|i| ((i % 211) as f64 - 100.0) * 0.01).collect();
        let k: Vec<f64> = (0..n).map(|i| ((i % 173) as f64 - 80.0) * 0.01).collect();
        // cos/sin as trailing [S, D] tile (already duplicated to head_dim).
        let cos: Vec<f64> = (0..tile)
            .map(|i| ((i * 7 % 100) as f64 * 0.01).cos())
            .collect();
        let sin: Vec<f64> = (0..tile)
            .map(|i| ((i * 7 % 100) as f64 * 0.01).sin())
            .collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let qt = sess
            .tensor_variable(q.clone(), vec![b, h, s, d], false)
            .unwrap();
        let kt = sess
            .tensor_variable(k.clone(), vec![b, h, s, d], false)
            .unwrap();
        let ct = sess
            .tensor_variable(cos.clone(), vec![s, d], false)
            .unwrap();
        let st = sess
            .tensor_variable(sin.clone(), vec![s, d], false)
            .unwrap();
        let (qe, ke) = sess.apply_rotary_pos_emb(qt, kt, ct, st, None).unwrap();
        let new_q = sess.tensor_values(qe).unwrap();
        let new_k = sess.tensor_values(ke).unwrap();
        let bitmatch = new_q == old_rope(&q, &cos, &sin, d, tile)
            && new_k == old_rope(&k, &cos, &sin, d, tile);

        let old_ms = bench(|| {
            old_rope(&q, &cos, &sin, d, tile).len() + old_rope(&k, &cos, &sin, d, tile).len()
        });
        let new_ms = bench(|| {
            let (qe, ke) = sess.apply_rotary_pos_emb(qt, kt, ct, st, None).unwrap();
            sess.tensor_values(qe).unwrap().len() + sess.tensor_values(ke).unwrap().len()
        });
        println!(
            "  {label:<20} ({:>3}MB q)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            n * 8 / (1 << 20),
            old_ms,
            new_ms,
            old_ms / new_ms,
            bitmatch
        );
    }
}
