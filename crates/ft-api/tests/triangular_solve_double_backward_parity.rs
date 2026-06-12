//! Regression: triangular_solve + cholesky_solve 1st- AND 2nd-order (Hessian)
//! parity vs PyTorch 2.12. tensor_triangular_solve was an `apply_function` custom
//! op with no create_graph backward, so functional_hessian of it (or of
//! cholesky_solve, which composes through it ×2) ERRORED. frankentorch-nj7x8:
//! added a cg backward that rebuilds grad_B = A⁻ᵀ·grad_X and the triangle-masked
//! grad_A from the differentiable general inverse (A⁻¹B == triangular solve for a
//! triangular A; only the mask is triangular-specific).
//!
//! Goldens: torch 2.12.0+cpu (f64).

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

// Lower-triangular A and rhs b.
const A_LOWER: [f64; 4] = [2.0, 0.0, 0.5, 1.5];
const B: [f64; 2] = [1.0, 2.0];

#[test]
fn triangular_solve_forward_and_first_grad_match_torch() {
    // forward: solve_triangular(A, b, upper=False) = [0.5, 1.166666667]
    let mut s = sess();
    let a = s.tensor_variable(A_LOWER.to_vec(), vec![2, 2], false).unwrap();
    let b = s.tensor_variable(B.to_vec(), vec![2], false).unwrap();
    let x = s.tensor_triangular_solve(a, b, false).unwrap();
    let xv = s.tensor_values(x).unwrap();
    close("tri fwd", &xv, &[0.5, 1.166666667]);

    // 1st-order grad wrt A (upper entry A[0,1] is masked -> 0)
    let mut s = sess();
    let a = s.tensor_variable(A_LOWER.to_vec(), vec![2, 2], true).unwrap();
    let b = s.tensor_variable(B.to_vec(), vec![2], false).unwrap();
    let x = s.tensor_triangular_solve(a, b, false).unwrap();
    let out = s.tensor_sum(x).unwrap();
    let rep = s.tensor_backward(out).unwrap();
    close(
        "tri grad_A",
        s.tensor_gradient(&rep, a).unwrap(),
        &[-0.166666667, 0.0, -0.333333333, -0.777777778],
    );
}

/// 2nd-order: full Hessian of sum(triangular_solve(A,b)) wrt A. The masked upper
/// entry gives a zero row/column — a sharp check of the triangle mask + the
/// differentiable-inverse compose.
#[test]
fn triangular_solve_hessian_over_a_matches_torch() {
    let mut s = sess();
    let a = s.tensor_variable(A_LOWER.to_vec(), vec![2, 2], true).unwrap();
    let b = s.tensor_variable(B.to_vec(), vec![2], false).unwrap();
    let x = s.tensor_triangular_solve(a, b, false).unwrap();
    let out = s.tensor_sum(x).unwrap();
    let h = s.tensor_functional_hessian(out, a).unwrap();
    let want = [
        0.166667, 0.0, 0.166667, -0.055556, //
        0.0, 0.0, 0.0, 0.0, //
        0.166667, 0.0, 0.0, 0.222222, //
        -0.055556, 0.0, 0.222222, 1.037037,
    ];
    close("tri hess A", &h, &want);
}

/// cholesky_solve composes through triangular_solve ×2. Over b it is LINEAR, so
/// its Hessian is exactly 0 — previously this ERRORED (no cg backward). (Over the
/// factor L, ft's gradient is masked to the triangle while torch's cholesky_solve
/// returns a DENSE factor gradient — a separate pre-existing 1st-order convention
/// difference, filed under frankentorch-nj7x8; not exercised here.)
#[test]
fn cholesky_solve_over_b_hessian_is_zero() {
    let mut s = sess();
    let lvals = vec![1.414213562, 0.0, 0.353553391, 1.17260394];
    let l = s.tensor_variable(lvals, vec![2, 2], false).unwrap();
    let b = s.tensor_variable(vec![1.0, 2.0], vec![2, 1], true).unwrap();
    let x = s.tensor_cholesky_solve(b, l, false).unwrap();
    let out = s.tensor_sum(x).unwrap();
    let h = s.tensor_functional_hessian(out, b).unwrap();
    close("cholsolve b hess", &h, &[0.0, 0.0, 0.0, 0.0]);
}
