//! A/B for tensor_embedding_bag F64. OLD = apply_function-path replica (CLONE the weight table via
//! to_vec — matches the orig's `inputs` materialization — then parallel gather+reduce); NEW =
//! sess.tensor_embedding_bag (F64 native fast path, borrow the table zero-copy). bitmatch verifies
//! the fast path matches. Run: cargo run --release -p ft-api --example embedding_bag_ab

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use rayon::prelude::*;
use std::time::Instant;

fn old_embedding_bag(
    weight: &[f64],
    input: &[f64],
    offsets: &[f64],
    dim: usize,
    num_bags: usize,
    num_emb: usize,
) -> Vec<f64> {
    let cloned = weight.to_vec(); // orig materializes the whole table via tensor_values
    let input_len = input.len();
    let bags: Vec<Vec<f64>> = (0..num_bags)
        .into_par_iter()
        .map(|b| {
            let start = offsets[b] as usize;
            let end = if b + 1 < num_bags { offsets[b + 1] as usize } else { input_len };
            let mut agg = vec![0.0_f64; dim];
            for &iv in input.iter().take(end.min(input_len)).skip(start) {
                let idx = (iv as usize).min(num_emb - 1);
                let es = idx * dim;
                for j in 0..dim {
                    agg[j] += cloned[es + j];
                }
            }
            agg
        })
        .collect();
    bags.into_iter().flatten().collect()
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
    println!("tensor_embedding_bag f64 sum, min-9:  OLD=apply_fn replica (clone table)  NEW=native (borrow)");
    // (label, num_embeddings, dim, num_bags, bag_size)
    let cases: [(&str, usize, usize, usize, usize); 3] = [
        ("500K x 64, 8K bags x4", 500_000, 64, 8_000, 4),
        ("1M x 32, 4K bags x8", 1_000_000, 32, 4_000, 8),
        ("250K x 128, 16K bags x2", 250_000, 128, 16_000, 2),
    ];
    for (label, num_emb, dim, num_bags, bag_size) in cases {
        let table_mb = num_emb * dim * 8 / (1 << 20);
        let weight: Vec<f64> = (0..num_emb * dim).map(|i| (i % 251) as f64 * 0.01).collect();
        let input_len = num_bags * bag_size;
        let input: Vec<f64> = (0..input_len).map(|i| ((i * 2_654_435_761usize) % num_emb) as f64).collect();
        let offsets: Vec<f64> = (0..num_bags).map(|b| (b * bag_size) as f64).collect();

        let mut sess = FrankenTorchSession::new(ExecutionMode::Strict);
        let wt = sess.tensor_variable(weight.clone(), vec![num_emb, dim], false).unwrap();
        let it = sess.tensor_variable(input.clone(), vec![input_len], false).unwrap();
        let ot = sess.tensor_variable(offsets.clone(), vec![num_bags], false).unwrap();
        let out = sess.tensor_embedding_bag(it, wt, ot, "sum").unwrap();
        let new_out = sess.tensor_values(out).unwrap();
        let old_out = old_embedding_bag(&weight, &input, &offsets, dim, num_bags, num_emb);
        let bitmatch = new_out == old_out;

        let old_ms = bench(|| old_embedding_bag(&weight, &input, &offsets, dim, num_bags, num_emb).len());
        let new_ms = bench(|| sess.tensor_embedding_bag(it, wt, ot, "sum").unwrap().0);
        println!(
            "  {label:<26} (table {:>4}MB)  OLD {:8.3}  NEW {:8.3}  = {:.2}x  bitmatch={}",
            table_mb, old_ms, new_ms, old_ms / new_ms, bitmatch
        );
    }
}
