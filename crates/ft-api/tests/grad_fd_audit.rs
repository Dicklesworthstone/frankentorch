//! Finite-difference gradient audit of ft-api backward paths (frankentorch-tyun0).
//!
//! Motivated by the addmm grad_mat1 transpose bug (f4c34512): gradient bugs of the
//! transpose/index class are masked by symmetric/square test inputs but exposed by
//! NON-square, NON-symmetric ones. These integration tests validate the FUSED linear
//! backward (`tensor_linear` -> `linear_backward_f64`, the most-used training op and
//! separate code from the now-fixed ft-autograd addmm) and the matmul backward
//! against numerical central differences — the gold-standard, analytic-expectation-
//! independent oracle. They go through the public `FrankenTorchSession` API so they
//! don't touch the (separately-owned) ft-api/ft-kernel-cpu source.

use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::ExecutionMode;

/// Validate the analytic gradient of `build_loss` w.r.t. `input` against numerical
/// central differences. `build_loss` takes the session and the (grad-tracked) input
/// node and returns a scalar loss node.
fn fd_check<F>(input: &[f64], shape: &[usize], build_loss: F)
where
    F: Fn(&mut FrankenTorchSession, TensorNodeId) -> TensorNodeId,
{
    let eps = 1e-6;
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s
        .tensor_variable(input.to_vec(), shape.to_vec(), true)
        .expect("input variable");
    let loss = build_loss(&mut s, x);
    s.tensor_backward(loss).expect("backward");
    let analytic = s
        .tensor_grad(x)
        .expect("grad lookup")
        .expect("input has a gradient");
    assert_eq!(analytic.len(), input.len(), "gradient length mismatch");

    let eval = |vals: &[f64]| -> f64 {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let xn = s
            .tensor_variable(vals.to_vec(), shape.to_vec(), false)
            .expect("eval variable");
        let l = build_loss(&mut s, xn);
        s.tensor_values(l).expect("loss value")[0]
    };
    for i in 0..input.len() {
        let mut up = input.to_vec();
        up[i] += eps;
        let mut dn = input.to_vec();
        dn[i] -= eps;
        let fd = (eval(&up) - eval(&dn)) / (2.0 * eps);
        assert!(
            (analytic[i] - fd).abs() < 1e-4,
            "grad[{i}]: analytic {} != finite-diff {} (diff {})",
            analytic[i],
            fd,
            (analytic[i] - fd).abs()
        );
    }
}

#[test]
fn linear_backward_input_grad_matches_finite_diff_nonsquare() {
    // y = x @ W^T + b. x[2,3] (batch=2,in=3), W[2,3] (out=2,in=3), b[2]. in != out,
    // non-symmetric W — the case class that exposed the addmm transpose bug.
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    let b = vec![0.1, 0.2];
    fd_check(&x, &[2, 3], move |s, xn| {
        let wn = s.tensor_variable(w.clone(), vec![2, 3], false).unwrap();
        let bn = s.tensor_variable(b.clone(), vec![2], false).unwrap();
        let y = s.tensor_linear(xn, wn, Some(bn)).unwrap();
        let z = s.tensor_mul(y, y).unwrap();
        s.tensor_sum(z).unwrap()
    });
}

#[test]
fn linear_backward_weight_grad_matches_finite_diff_nonsquare() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    let b = vec![0.1, 0.2];
    fd_check(&w, &[2, 3], move |s, wn| {
        let xn = s.tensor_variable(x.clone(), vec![2, 3], false).unwrap();
        let bn = s.tensor_variable(b.clone(), vec![2], false).unwrap();
        let y = s.tensor_linear(xn, wn, Some(bn)).unwrap();
        let z = s.tensor_mul(y, y).unwrap();
        s.tensor_sum(z).unwrap()
    });
}

#[test]
fn linear_backward_bias_grad_matches_finite_diff() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let w = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    fd_check(&[0.1, 0.2], &[2], move |s, bn| {
        let xn = s.tensor_variable(x.clone(), vec![2, 3], false).unwrap();
        let wn = s.tensor_variable(w.clone(), vec![2, 3], false).unwrap();
        let y = s.tensor_linear(xn, wn, Some(bn)).unwrap();
        let z = s.tensor_mul(y, y).unwrap();
        s.tensor_sum(z).unwrap()
    });
}

#[test]
fn matmul_backward_grads_match_finite_diff_nonsquare() {
    // A[2,3] @ B[3,2] -> [2,2]; non-square, non-symmetric. Check both operands.
    let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
    let b1 = b.clone();
    fd_check(&a, &[2, 3], move |s, an| {
        let bn = s.tensor_variable(b1.clone(), vec![3, 2], false).unwrap();
        let out = s.tensor_matmul(an, bn).unwrap();
        let z = s.tensor_mul(out, out).unwrap();
        s.tensor_sum(z).unwrap()
    });
    let a1 = a.clone();
    fd_check(&b, &[3, 2], move |s, bn| {
        let an = s.tensor_variable(a1.clone(), vec![2, 3], false).unwrap();
        let out = s.tensor_matmul(an, bn).unwrap();
        let z = s.tensor_mul(out, out).unwrap();
        s.tensor_sum(z).unwrap()
    });
}

// A non-symmetric, diagonally-dominant (well-conditioned, invertible) 3x3 matrix.
// Non-symmetric is the key: the inv/det/solve/expm adjoints are transpose-heavy
// (e.g. d(inv)/dA involves inv(A)^T), so a transpose bug there is invisible to
// symmetric tests but caught here.
fn nonsym_3x3() -> Vec<f64> {
    vec![2.0, 0.5, 0.1, 0.3, 3.0, 0.2, 0.1, 0.4, 4.0]
}

#[test]
fn inverse_backward_grad_matches_finite_diff_nonsymmetric() {
    fd_check(&nonsym_3x3(), &[3, 3], |s, an| {
        let inv = s.tensor_inverse(an).unwrap();
        let z = s.tensor_mul(inv, inv).unwrap();
        s.tensor_sum(z).unwrap()
    });
}

#[test]
fn det_backward_grad_matches_finite_diff_nonsymmetric() {
    // d(det A)/dA = det(A) * inv(A)^T — transpose in the adjoint.
    fd_check(&nonsym_3x3(), &[3, 3], |s, an| s.tensor_det(an).unwrap());
}

#[test]
fn matrix_exp_backward_grad_matches_finite_diff_nonsymmetric() {
    // scale down so expm is well-conditioned for finite differences
    let a: Vec<f64> = nonsym_3x3().iter().map(|v| v * 0.1).collect();
    fd_check(&a, &[3, 3], |s, an| {
        let e = s.tensor_matrix_exp(an).unwrap();
        let z = s.tensor_mul(e, e).unwrap();
        s.tensor_sum(z).unwrap()
    });
}

#[test]
fn solve_backward_grad_wrt_a_matches_finite_diff_nonsymmetric() {
    // x = A^{-1} b; the adjoint w.r.t. A carries A^{-T} — transpose-heavy.
    let b = vec![1.0, 2.0, 3.0];
    fd_check(&nonsym_3x3(), &[3, 3], move |s, an| {
        let bn = s.tensor_variable(b.clone(), vec![3, 1], false).unwrap();
        let x = s.tensor_solve(an, bn).unwrap();
        let z = s.tensor_mul(x, x).unwrap();
        s.tensor_sum(z).unwrap()
    });
}

#[test]
fn qr_backward_r_grad_matches_finite_diff_nonsymmetric() {
    // QR adjoint involves a copyltu/triangular-transpose step — transpose-heavy.
    // Check d(sum(R^2))/dA on a non-symmetric square matrix.
    fd_check(&nonsym_3x3(), &[3, 3], |s, an| {
        let (_q, r) = s.tensor_qr(an, true).unwrap();
        let z = s.tensor_mul(r, r).unwrap();
        s.tensor_sum(z).unwrap()
    });
}

// ── Symmetric-domain linalg (eigvalsh / eigh / cholesky) ───────────────────
// These ops require a symmetric (and for cholesky, PD) input, so perturbing a
// single entry breaks the domain. Instead build the symmetric input A = P + P^T
// from a FREE parameter P (cholesky adds c*I for positive-definiteness): FD over P
// is unconstrained and validates the op's adjoint composed with the already-audited
// add + transpose backwards — convention-agnostic w.r.t. how the op symmetrizes its
// own gradient.

#[test]
fn eigvalsh_backward_grad_matches_finite_diff() {
    // A = P + P^T (symmetric, well-separated eigenvalues -> smooth eigvalsh).
    let p = vec![2.0, 0.5, 0.1, 0.3, 3.0, 0.2, 0.1, 0.4, 4.0];
    fd_check(&p, &[3, 3], |s, pn| {
        let pt = s.tensor_transpose(pn, 0, 1).unwrap();
        let a = s.tensor_add(pn, pt).unwrap();
        let ev = s.tensor_eigvalsh(a).unwrap();
        let z = s.tensor_mul(ev, ev).unwrap();
        s.tensor_sum(z).unwrap()
    });
}

#[test]
fn eigh_eigenvalue_backward_grad_matches_finite_diff() {
    // Same symmetric construction; loss over eigenVALUES only (eigenvectors carry a
    // sign/gauge ambiguity that makes their gradient ill-defined for FD).
    let p = vec![2.0, 0.5, 0.1, 0.3, 3.0, 0.2, 0.1, 0.4, 4.0];
    fd_check(&p, &[3, 3], |s, pn| {
        let pt = s.tensor_transpose(pn, 0, 1).unwrap();
        let a = s.tensor_add(pn, pt).unwrap();
        let (vals, _vecs) = s.tensor_eigh(a).unwrap();
        let z = s.tensor_mul(vals, vals).unwrap();
        s.tensor_sum(z).unwrap()
    });
}

#[test]
fn cholesky_backward_grad_matches_finite_diff() {
    // A = P + P^T + 5*I -> symmetric positive-definite (diagonally dominant); FD over
    // the free P validates the cholesky adjoint (the tricky lower-triangular Phi op).
    let p = vec![0.5, 0.2, 0.1, 0.3, 0.4, 0.15, 0.1, 0.25, 0.6];
    fd_check(&p, &[3, 3], |s, pn| {
        let pt = s.tensor_transpose(pn, 0, 1).unwrap();
        let sym = s.tensor_add(pn, pt).unwrap();
        let cid = s
            .tensor_variable(
                vec![5.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 5.0],
                vec![3, 3],
                false,
            )
            .unwrap();
        let a = s.tensor_add(sym, cid).unwrap();
        let l = s.tensor_cholesky(a, false).unwrap();
        let z = s.tensor_mul(l, l).unwrap();
        s.tensor_sum(z).unwrap()
    });
}

// ── Fused backward kernels ─────────────────────────────────────────────────
// These (SDPA, conv2d, layer_norm) are the highest remaining addmm-class bug risk:
// fused custom-op backwards full of matmuls/transposes/reductions, on general
// (non-symmetric) inputs so finite differences are clean.

#[test]
fn sdpa_backward_grads_match_finite_diff() {
    // Q/K/V [bh=1, seq=3, dim=2]. The fused sdpa_backward is matmul/transpose-heavy:
    // dV=P^T@dOut, dP=dOut@V^T, dQ=scale*dU@K, dK=scale*dU^T@Q.
    let q = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
    let k = vec![0.2, 0.1, 0.05, 0.3, 0.15, 0.25];
    let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let sh = vec![1usize, 3, 2];
    let loss = |s: &mut FrankenTorchSession, qn, kn, vn| {
        let o = s
            .tensor_scaled_dot_product_attention(qn, kn, vn, None, false, None)
            .unwrap();
        let z = s.tensor_mul(o, o).unwrap();
        s.tensor_sum(z).unwrap()
    };
    let (k1, v1, sh1) = (k.clone(), v.clone(), sh.clone());
    fd_check(&q, &sh, move |s, qn| {
        let kn = s.tensor_variable(k1.clone(), sh1.clone(), false).unwrap();
        let vn = s.tensor_variable(v1.clone(), sh1.clone(), false).unwrap();
        loss(s, qn, kn, vn)
    });
    let (q1, v2, sh2) = (q.clone(), v.clone(), sh.clone());
    fd_check(&k, &sh, move |s, kn| {
        let qn = s.tensor_variable(q1.clone(), sh2.clone(), false).unwrap();
        let vn = s.tensor_variable(v2.clone(), sh2.clone(), false).unwrap();
        loss(s, qn, kn, vn)
    });
    let (q2, k2, sh3) = (q.clone(), k.clone(), sh.clone());
    fd_check(&v, &sh, move |s, vn| {
        let qn = s.tensor_variable(q2.clone(), sh3.clone(), false).unwrap();
        let kn = s.tensor_variable(k2.clone(), sh3.clone(), false).unwrap();
        loss(s, qn, kn, vn)
    });
}

#[test]
fn conv2d_backward_grads_match_finite_diff() {
    // input[1,1,3,3] * weight[1,1,2,2] -> [1,1,2,2]. Weight gradient (correlation /
    // transpose of the kernel windows) is the addmm-class risk.
    let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let weight = vec![0.5, -0.3, 0.2, 0.7];
    let w1 = weight.clone();
    fd_check(&input, &[1, 1, 3, 3], move |s, xn| {
        let wn = s
            .tensor_variable(w1.clone(), vec![1, 1, 2, 2], false)
            .unwrap();
        let o = s.tensor_conv2d(xn, wn, None, (1, 1), (0, 0)).unwrap();
        let z = s.tensor_mul(o, o).unwrap();
        s.tensor_sum(z).unwrap()
    });
    let i1 = input.clone();
    fd_check(&weight, &[1, 1, 2, 2], move |s, wn| {
        let xn = s
            .tensor_variable(i1.clone(), vec![1, 1, 3, 3], false)
            .unwrap();
        let o = s.tensor_conv2d(xn, wn, None, (1, 1), (0, 0)).unwrap();
        let z = s.tensor_mul(o, o).unwrap();
        s.tensor_sum(z).unwrap()
    });
}

#[test]
fn layer_norm_backward_grads_match_finite_diff() {
    // input[2,3] normalized over [3], affine weight[3] + bias[3].
    let input = vec![1.0, 2.0, 4.0, -1.0, 0.5, 3.0];
    let weight = vec![1.1, 0.9, 1.3];
    let bias = vec![0.2, -0.1, 0.05];
    let (w1, b1) = (weight.clone(), bias.clone());
    fd_check(&input, &[2, 3], move |s, xn| {
        let wn = s.tensor_variable(w1.clone(), vec![3], false).unwrap();
        let bn = s.tensor_variable(b1.clone(), vec![3], false).unwrap();
        let o = s
            .tensor_layer_norm(xn, vec![3], Some(wn), Some(bn), 1e-5)
            .unwrap();
        let z = s.tensor_mul(o, o).unwrap();
        s.tensor_sum(z).unwrap()
    });
    let (i1, b2) = (input.clone(), bias.clone());
    fd_check(&weight, &[3], move |s, wn| {
        let xn = s.tensor_variable(i1.clone(), vec![2, 3], false).unwrap();
        let bn = s.tensor_variable(b2.clone(), vec![3], false).unwrap();
        let o = s
            .tensor_layer_norm(xn, vec![3], Some(wn), Some(bn), 1e-5)
            .unwrap();
        let z = s.tensor_mul(o, o).unwrap();
        s.tensor_sum(z).unwrap()
    });
}
