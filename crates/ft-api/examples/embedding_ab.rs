//! A/B for tensor_embedding F64. OLD = replica of the pre-fix path (CLONE the weight table via to_vec
//! — matches the old `tensor_values(weight)` — then parallel per-row gather); NEW = sess.tensor_embedding
//! (F64 borrows the table zero-copy). bitmatch verifies the borrow path matches.
//! Run: cargo run --release -p ft-api --example embedding_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use rayon::prelude::*;
use std::time::Instant;

fn old_embedding(weight: &[f64], idx: &[usize], dim: usize) -> Vec<f64> {
    let cloned = weight.to_vec(); // old path materialized the whole table via tensor_values
    let mut result = vec![0.0_f64; idx.len() * dim];
    result
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(i, row)| {
            let s = idx[i] * dim;
            row.copy_from_slice(&cloned[s..s + dim]);
        });
    result
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
    println!("tensor_embedding f64, min-9:  OLD=clone table + gather  NEW=borrow table + gather");
    // (label, num_embeddings, dim, num_indices)
    let cases: [(&str, usize, usize, usize); 3] = [
        ("50K x 768, 16K lookups", 50_000, 768, 16_000),
        ("250K x 128, 8K lookups", 250_000, 128, 8_000),
        ("1M x 64, 32K lookups", 1_000_000, 64, 32_000),
    ];
    for (label, num_emb, dim, num_idx) in cases {
        let table_mb = num_emb * dim * 8 / (1 << 20);
        let weight: Vec<f64> = (0..num_emb * dim).map(|i| (i % 251) as f64 * 0.01).collect();
        let idx_usize: Vec<usize> = (0..num_idx).map(|i| (i * 2_654_435_761usize) % num_emb).collect();
        let idx_f64: Vec<f64> = idx_usize.iter().map(|&i| i as f64).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let wt = sess.tensor_variable(weight.clone(), vec![num_emb, dim], false).unwrap();
        let it = sess.tensor_variable(idx_f64.clone(), vec![num_idx], false).unwrap();
        let out = sess.tensor_embedding(it, wt, None).unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_embedding(&weight, &idx_usize, dim);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_embedding(&weight, &idx_usize, dim).len());
        let new_ms = bench(|| sess.tensor_embedding(it, wt, None).unwrap().0);
        println!(
            "  {label:<24} (table {:>4}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            table_mb, old_ms, new_ms, old_ms / new_ms, bitmatch
        );
    }
}
