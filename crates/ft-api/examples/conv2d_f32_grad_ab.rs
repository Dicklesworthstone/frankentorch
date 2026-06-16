//! End-to-end A/B for the f32 conv2d GRAD fast path (frankentorch-48w0b). OLD =
//! the manual composed conv2d (unfold + permute + reshape + matmul, the path f32
//! conv grad took before the fast path) backward; NEW = functional_conv2d (which
//! now takes the fused f32-output custom-op grad fast path) backward. Fresh
//! session per iteration (the tape never frees nodes — gmuml). Min-time per arm.
//!   cargo run -q --release -p ft-api --example conv2d_f32_grad_ab
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let (n, cin, cout, ih, iw, k) = (8usize, 64usize, 64usize, 30usize, 30usize, 3usize);
    let (sh, sw, ph, pw) = (1usize, 1usize, 1usize, 1usize);
    let xv: Vec<f32> = (0..n * cin * ih * iw).map(|i| (i % 877) as f32 * 0.01).collect();
    let wv: Vec<f32> = (0..cout * cin * k * k).map(|i| (i % 47) as f32 * 0.1 - 2.0).collect();
    let oh = (ih + 2 * ph - k) / sh + 1;
    let ow = (iw + 2 * pw - k) / sw + 1;
    let patch_w = cin * k * k;
    let patch_c = oh * ow;

    // NEW: fused fast path.
    let new_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(xv.clone(), vec![n, cin, ih, iw], true).unwrap();
        let w = s.tensor_variable_f32(wv.clone(), vec![cout, cin, k, k], true).unwrap();
        let o = s.functional_conv2d(x, w, None, (sh, sw), (ph, pw)).unwrap();
        let l = s.tensor_sum(o).unwrap();
        s.tensor_backward(l).unwrap();
    };
    // OLD: manual composed conv2d (unfold+permute+reshape+matmul).
    let old_step = || {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(xv.clone(), vec![n, cin, ih, iw], true).unwrap();
        let w = s.tensor_variable_f32(wv.clone(), vec![cout, cin, k, k], true).unwrap();
        let padded = s.tensor_pad(x, &[pw, pw, ph, ph], 0.0).unwrap();
        let uh = s.tensor_unfold(padded, 2, k, sh).unwrap();
        let uhw = s.tensor_unfold(uh, 3, k, sw).unwrap();
        let perm = s.tensor_permute(uhw, vec![0, 2, 3, 1, 4, 5]).unwrap();
        let unf = s.tensor_reshape(perm, vec![n, patch_c, patch_w]).unwrap();
        let wf = s.tensor_reshape(w, vec![cout, patch_w]).unwrap();
        let wt = s.tensor_transpose(wf, 0, 1).unwrap();
        let uflat = s.tensor_reshape(unf, vec![n * patch_c, patch_w]).unwrap();
        let of = s.tensor_matmul(uflat, wt).unwrap();
        let o = s.tensor_reshape(of, vec![n, patch_c, cout]).unwrap();
        let o = s.tensor_transpose(o, 1, 2).unwrap();
        let o = s.tensor_reshape(o, vec![n, cout, oh, ow]).unwrap();
        let l = s.tensor_sum(o).unwrap();
        s.tensor_backward(l).unwrap();
    };

    new_step();
    old_step();
    let reps = 15;
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
    eprintln!("conv2d f32 fwd+bwd [{n},{cin}->{cout},{ih}x{iw}] k{k}: composed {bo:.2} ms / fused {bn:.2} ms / speedup {:.2}x", bo / bn);
}
