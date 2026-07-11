//! Regression: cholesky decomposition 1st- AND 2nd-order (Hessian) parity vs
//! PyTorch 2.12. tensor_cholesky was an `apply_function` custom op with no
//! create_graph backward, so functional_hessian of a cholesky-containing loss
//! ERRORED. frankentorch-nj7x8: added a cg backward rebuilding the differentiable
//! Cholesky VJP — phi = Φ(Lᵀ·grad_L); grad_A = sym(L⁻ᵀ·phi·L⁻¹) — from a
//! differentiable cholesky node + the differentiable inverse, with Φ (tril, halved
//! diagonal) as a constant mask multiply.
//!
//! Goldens: torch 2.12.0+cpu (f64), A = [[2,0.5],[0.5,1.5]].

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-6,
            "{label}[{i}]: got {g}, want {w} (torch)"
        );
    }
}

fn sess() -> FrankenTorchSession {
    FrankenTorchSession::new(ExecutionMode::Strict)
}

const A: [f64; 4] = [2.0, 0.5, 0.5, 1.5];

// torch full 4x4 Hessian of sum(cholesky(A)) (lower == upper here).
const CHOL_HESS: [f64; 16] = [
    -0.049353, -0.032665, -0.032665, -0.009691, //
    -0.032665, -0.116291, -0.116291, 0.038764, //
    -0.032665, -0.116291, -0.116291, 0.038764, //
    -0.009691, 0.038764, 0.038764, -0.155055,
];

#[test]
fn cholesky_forward_and_first_grad_match_torch() {
    // forward (lower): L = [[1.41421, 0],[0.35355, 1.17260]]
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], false).unwrap();
    let l = s.tensor_cholesky(m, false).unwrap();
    let lv = s.tensor_values(l).unwrap();
    close(
        "L fwd",
        &lv,
        &[std::f64::consts::SQRT_2, 0.0, 0.353553391, 1.17260394],
    );

    // 1st-order grad of sum(L) wrt A
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], true).unwrap();
    let l = s.tensor_cholesky(m, false).unwrap();
    let out = s.tensor_sum(l).unwrap();
    let rep = s.tensor_backward(out).unwrap();
    close(
        "L grad_A",
        s.tensor_gradient(&rep, m).unwrap(),
        &[0.291815132, 0.246953032, 0.246953032, 0.426401433],
    );
}

/// 2nd-order: full Hessian of sum(cholesky(A)) wrt A (lower factor).
#[test]
fn cholesky_lower_hessian_matches_torch() {
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], true).unwrap();
    let l = s.tensor_cholesky(m, false).unwrap();
    let out = s.tensor_sum(l).unwrap();
    let h = s.tensor_functional_hessian(out, m).unwrap();
    close("chol lower hess", &h, &CHOL_HESS);
}

/// 2nd-order: upper factor — exercises the U = Lᵀ transpose path in the cg backward.
#[test]
fn cholesky_upper_hessian_matches_torch() {
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], true).unwrap();
    let u = s.tensor_cholesky(m, true).unwrap();
    let out = s.tensor_sum(u).unwrap();
    let h = s.tensor_functional_hessian(out, m).unwrap();
    close("chol upper hess", &h, &CHOL_HESS);
}
