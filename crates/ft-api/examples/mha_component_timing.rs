//! Localize the MHA f64 inference overhead (BlackThrush): MHA = ~10 composed ops and
//! ran 145ms vs PyTorch 25ms (~20x the ~10ms compute). Is it ONE slow primitive
//! (matmul / permute = a BROAD lever, used everywhere) or distributed per-op node/clone
//! overhead (architectural)? Time each component op at MHA scale, no-grad, min.
//!
//! Run: cargo run --release -p ft-api --example mha_component_timing

use std::time::Instant;

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const SEQ: usize = 256;
const BATCH: usize = 8;
const EMBED: usize = 512;
const HEADS: usize = 8;
const D: usize = EMBED / HEADS;

fn vals(n: usize, shift: f64) -> Vec<f64> {
    (0..n)
        .map(|i| (((i as f64) * 0.011 + shift).sin()) * 0.1)
        .collect()
}

fn timed(label: &str, iters: usize, mut f: impl FnMut()) {
    for _ in 0..3 {
        f();
    }
    let mut t = Vec::with_capacity(iters);
    for _ in 0..iters {
        let a = Instant::now();
        f();
        t.push(a.elapsed().as_secs_f64() * 1e3);
    }
    t.sort_by(|a, b| a.partial_cmp(b).unwrap());
    println!("  {label:34} {:8.3} ms (min)", t[0]);
}

fn main() {
    let iters: usize = std::env::var("ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);
    let m = SEQ * BATCH; // 2048 rows
    let a_proj = vals(m * EMBED, 0.0); // [2048,512]
    let w_proj = vals(EMBED * EMBED, 0.3); // [512,512]
    let qkv = vals(m * EMBED, 0.1); // [seq*batch, embed] for reshape/permute

    println!(
        "MHA component ops at scale [seq={SEQ},batch={BATCH},embed={EMBED},heads={HEADS}], {iters} iters MIN:"
    );

    // 1) one in-proj matmul [2048,512] @ [512,512]
    timed("matmul [2048,512]@[512,512]", iters, || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(a_proj.clone(), vec![m, EMBED], false)
            .unwrap();
        let w = s
            .tensor_variable(w_proj.clone(), vec![EMBED, EMBED], false)
            .unwrap();
        let o = s.tensor_matmul(a, w).unwrap();
        std::hint::black_box(s.tensor_values(o).unwrap());
    });

    // 2) reshape [seq,batch,embed] -> [seq,batch,heads,d] (view-ish)
    timed("reshape ->[seq,batch,heads,d]", iters, || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(qkv.clone(), vec![SEQ, BATCH, EMBED], false)
            .unwrap();
        let o = s.tensor_reshape(x, vec![SEQ, BATCH, HEADS, D]).unwrap();
        std::hint::black_box(s.tensor_values(o).unwrap());
    });

    // 3) permute to [batch,heads,seq,d] (the head-major transpose)
    timed("permute ->[batch,heads,seq,d]", iters, || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable(qkv.clone(), vec![SEQ, BATCH, HEADS, D], false)
            .unwrap();
        let o = s.tensor_permute(x, vec![1, 2, 0, 3]).unwrap();
        std::hint::black_box(s.tensor_values(o).unwrap());
    });

    // 4) session+leaf overhead alone (1 tensor_variable + read) at proj scale
    timed("session+1 leaf [2048,512]", iters, || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let a = s
            .tensor_variable(a_proj.clone(), vec![m, EMBED], false)
            .unwrap();
        std::hint::black_box(s.tensor_values(a).unwrap());
    });
}
