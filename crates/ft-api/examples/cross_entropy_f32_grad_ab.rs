//! End-to-end A/B for the f32 cross-entropy GRAD fast path (frankentorch-48w0b).
//! OLD = the composed op-graph (log_softmax over [batch,classes] + nll gather —
//! the path f32 cross-entropy grad took before; the f64 fused path existed but f32
//! fell through); NEW = functional_cross_entropy (fused f32-output custom op:
//! cross_entropy_forward_f32 streaming lse−logit[target] + cross_entropy_backward_f32
//! softmax−onehot, never materialising the [batch,classes] log-softmax nor its grad
//! tape). Fresh session/iter (tape never frees — gmuml). Min-time per arm.
//!   cargo run -q --release -p ft-api --example cross_entropy_f32_grad_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    // Language-model head: [batch*seq, vocab].
    let (batch, classes) = (8192usize, 4096usize);
    let logits: Vec<f32> = (0..batch * classes)
        .map(|i| ((i % 877) as f32 - 400.0) * 0.002)
        .collect();
    // Target holds class indices; nll_loss (the composed reference) reads it as
    // f64, so use an f64 target in both arms (the fused f32 path accepts either).
    let tgt: Vec<f64> = (0..batch).map(|i| (i % classes) as f64).collect();

    let new_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(logits.clone(), vec![batch, classes], true).unwrap();
        let t = s.tensor_variable(tgt.clone(), vec![batch], false).unwrap();
        let o = s.functional_cross_entropy(x, t, "mean").unwrap();
        s.tensor_backward(o).unwrap();
    };
    // OLD: composed log_softmax + nll_loss op-graph.
    let old_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(logits.clone(), vec![batch, classes], true).unwrap();
        let t = s.tensor_variable(tgt.clone(), vec![batch], false).unwrap();
        let lp = s.tensor_log_softmax(x, 1).unwrap();
        let o = s.tensor_nll_loss(lp, t, "mean").unwrap();
        s.tensor_backward(o).unwrap();
    };

    new_step();
    old_step();
    let reps = 10;
    let mut bo = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        old_step();
        bo = bo.min(t.elapsed().as_secs_f64() * 1e3);
    }
    let mut bn = f64::INFINITY;
    for _ in 0..reps {
        let t = Instant::now();
        new_step();
        bn = bn.min(t.elapsed().as_secs_f64() * 1e3);
    }
    eprintln!("cross_entropy f32 fwd+bwd [{batch}x{classes}]: composed {bo:.2} ms / fused {bn:.2} ms / speedup {:.2}x", bo / bn);
}
