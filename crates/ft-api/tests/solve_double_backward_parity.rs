//! Regression: linalg_solve 1st- AND 2nd-order (Hessian) parity vs PyTorch 2.12.
//! solve(A,b) composes through tensor_linalg_inv + tensor_matmul; once inverse
//! got a create_graph backward (frankentorch-b820k), solve's 2nd derivative
//! works "for free". This locks that propagation so a regression in the inverse
//! cg backward (or solve's compose path) fails loudly.
//!
//! Goldens: torch 2.12.0+cpu (f64), A=[[2,0.5],[0.5,1.5]], b=[1,2].

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
const B: [f64; 2] = [1.0, 2.0];

#[test]
fn solve_forward_and_first_grad_match_torch() {
    // forward: solve(A,b) = [0.181818182, 1.272727273]
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], false).unwrap();
    let bv = s.tensor_variable(B.to_vec(), vec![2], false).unwrap();
    let x = s.tensor_linalg_solve(m, bv).unwrap();
    let xv = s.tensor_values(x).unwrap();
    close("solve fwd", &xv, &[0.181818182, 1.272727273]);

    // 1st-order grad of sum(solve(A,b)) wrt A
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], true).unwrap();
    let bv = s.tensor_variable(B.to_vec(), vec![2], false).unwrap();
    let x = s.tensor_linalg_solve(m, bv).unwrap();
    let out = s.tensor_sum(x).unwrap();
    let rep = s.tensor_backward(out).unwrap();
    close(
        "solve.grad_A",
        s.tensor_gradient(&rep, m).unwrap(),
        &[-0.066115702, -0.462809917, -0.099173554, -0.694214876],
    );
}

/// 2nd-order: Hessian of sum(solve(A,b)) wrt A — works via the inverse cg backward.
#[test]
fn solve_hessian_matches_torch() {
    let mut s = sess();
    let m = s.tensor_variable(A.to_vec(), vec![2, 2], true).unwrap();
    let bv = s.tensor_variable(B.to_vec(), vec![2], false).unwrap();
    let x = s.tensor_linalg_solve(m, bv).unwrap();
    let out = s.tensor_sum(x).unwrap();
    let h = s.tensor_functional_hessian(out, m).unwrap();
    let diag: Vec<f64> = (0..4).map(|i| h[i * 4 + i]).collect();
    close(
        "solve hess",
        &diag,
        &[0.072126, -0.168295, -0.036063, 1.009767],
    );
    assert!((h[3] - (-0.102179)).abs() <= 1e-6, "H03 {}", h[3]);
}
