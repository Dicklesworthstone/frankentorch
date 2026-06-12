//! Regression: matrix inverse + logdet/slogdet 1st- AND 2nd-order (Hessian)
//! parity vs PyTorch 2.12. linalg_inv and slogdet were `apply_function` custom
//! ops with NO create_graph backward, so `torch.autograd.functional.hessian` of
//! any inverse/logdet-containing loss ERRORED. frankentorch-b820k: added a shared
//! differentiable-inverse cg node so the matrix-valued 2nd derivative propagates
//! (d logdet/dA = inv(A)ᵀ; d inv/dA = the -inv⊗inv tensor).
//!
//! Goldens generated with torch 2.12.0+cpu (f64), A = [[2,0.5],[0.5,1.5]].

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() <= 1e-6, "{label}[{i}]: got {g}, want {w} (torch)");
    }
}

fn sess() -> FrankenTorchSession {
    FrankenTorchSession::new(ExecutionMode::Strict)
}

const A: [f64; 4] = [2.0, 0.5, 0.5, 1.5];

/// 1st-order grads unchanged by the cg-wrapper swap (forward + backward kept).
#[test]
fn inverse_logdet_first_order_match_torch() {
    // inverse forward value
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], false).unwrap();
    let inv = s.tensor_inverse(m).unwrap();
    let invv = s.tensor_values(inv).unwrap();
    close(
        "inv",
        &invv,
        &[0.545454545, -0.181818182, -0.181818182, 0.727272727],
    );

    // logdet 1st-order grad = inv(A)ᵀ (symmetric here)
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], true).unwrap();
    let ld = s.tensor_logdet(m).unwrap();
    let rep = s.tensor_backward(ld).unwrap();
    close(
        "logdet.grad",
        s.tensor_gradient(&rep, m).unwrap(),
        &[0.545454545, -0.181818182, -0.181818182, 0.727272727],
    );

    // sum(inv(A)) 1st-order grad
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], true).unwrap();
    let inv = s.tensor_inverse(m).unwrap();
    let out = s.tensor_sum(inv).unwrap();
    let rep = s.tensor_backward(out).unwrap();
    close(
        "sum_inv.grad",
        s.tensor_gradient(&rep, m).unwrap(),
        &[-0.132231405, -0.198347107, -0.198347107, -0.297520661],
    );
}

/// 2nd-order: Hessian of logdet(A) (4x4 over the flattened 2x2 input).
#[test]
fn logdet_hessian_matches_torch() {
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], true).unwrap();
    let ld = s.tensor_logdet(m).unwrap();
    let h = s.tensor_functional_hessian(ld, m).unwrap();
    let diag: Vec<f64> = (0..4).map(|i| h[i * 4 + i]).collect();
    close(
        "logdet hess",
        &diag,
        &[-0.297521, -0.033058, -0.033058, -0.528926],
    );
    assert!((h[3] - (-0.033058)).abs() <= 1e-6, "H03 {}", h[3]);
}

/// 2nd-order: Hessian of trace(inv(A)) and sum(inv(A)) — exercises the inverse
/// cg backward (-Yᵀ·G·Yᵀ rebuilt from a differentiable inverse node).
#[test]
fn inverse_hessian_matches_torch() {
    // trace(inv(A))
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], true).unwrap();
    let inv = s.tensor_inverse(m).unwrap();
    let tr = s.tensor_trace(inv).unwrap();
    let h = s.tensor_functional_hessian(tr, m).unwrap();
    let diag: Vec<f64> = (0..4).map(|i| h[i * 4 + i]).collect();
    close(
        "trace_inv hess",
        &diag,
        &[0.360631, 0.084147, 0.084147, 0.817431],
    );

    // sum(inv(A))
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], true).unwrap();
    let inv = s.tensor_inverse(m).unwrap();
    let out = s.tensor_sum(inv).unwrap();
    let h = s.tensor_functional_hessian(out, m).unwrap();
    let diag: Vec<f64> = (0..4).map(|i| h[i * 4 + i]).collect();
    close(
        "sum_inv hess",
        &diag,
        &[0.144252, -0.072126, -0.072126, 0.432757],
    );
}

/// 2nd-order: det(A) via Jacobi's formula. For 2×2 det=ad-bc the Hessian has a
/// ZERO diagonal and ±1 off-diagonals (∂²det/∂a∂d=1, ∂²det/∂b∂c=-1) — a sharp
/// check that the det cg backward differentiates BOTH det(A) and inv(A).
#[test]
fn det_hessian_matches_torch() {
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], true).unwrap();
    let d = s.tensor_det(m).unwrap();
    // 1st-order grad = det·inv(A)ᵀ
    let mut s2 = sess();
    let m2 = s2.tensor_variable(A.to_vec(), vec![2, 2], true).unwrap();
    let d2 = s2.tensor_det(m2).unwrap();
    let rep = s2.tensor_backward(d2).unwrap();
    close("det.grad", s2.tensor_gradient(&rep, m2).unwrap(), &[1.5, -0.5, -0.5, 2.0]);
    // 2nd-order: full 4x4 Hessian = [[0,0,0,1],[0,0,-1,0],[0,-1,0,0],[1,0,0,0]]
    let h = s.tensor_functional_hessian(d, m).unwrap();
    let want = [
        0.0, 0.0, 0.0, 1.0, 0.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
    ];
    close("det hess", &h, &want);
}
