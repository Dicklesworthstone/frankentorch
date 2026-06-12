//! Regression: logsumexp forward + 1st-order grad + 2nd-order (Hessian) parity
//! vs PyTorch 2.12. logsumexp was an `apply_function` custom op with NO
//! create_graph backward, so `torch.autograd.functional.hessian` of any
//! logsumexp-containing loss ERRORED ("requires a create_graph backward").
//! frankentorch-t1q4: added the cg backward d/dx logsumexp(x,dim)=softmax(x,dim),
//! built from the differentiable tape softmax, so the 2nd derivative propagates.
//!
//! Goldens generated with torch 2.12.0+cpu (f64).

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        // goldens are torch values rounded to 6 decimals
        assert!((g - w).abs() <= 1e-6, "{label}[{i}]: got {g}, want {w} (torch)");
    }
}

fn sess() -> FrankenTorchSession {
    FrankenTorchSession::new(ExecutionMode::Strict)
}

/// Forward + 1st-order backward unchanged by the wrapper swap (dim=1 on [2,3]).
#[test]
fn logsumexp_forward_and_first_grad_match_torch() {
    let a = vec![0.5, 1.0, 1.5, 2.0, -1.0, 0.3];
    // forward: logsumexp([[.5,1,1.5],[2,-1,.3]], dim=1) = [2.18027, 2.209021]
    let mut s = sess();
    let x = s.tensor_variable(a.clone(), vec![2, 3], false).unwrap();
    let y = s.tensor_logsumexp(x, 1).unwrap();
    let fwd = s.tensor_values(y).unwrap();
    close("fwd", &fwd, &[2.18027, 2.209021]);

    // 1st-order: loss = sum(logsumexp(x,1) * [1,2]); grad matches torch.
    let mut s = sess();
    let x = s.tensor_variable(a, vec![2, 3], true).unwrap();
    let y = s.tensor_logsumexp(x, 1).unwrap();
    let w = s.tensor_variable(vec![1.0, 2.0], vec![2], false).unwrap();
    let prod = s.tensor_mul(y, w).unwrap();
    let out = s.tensor_sum(prod).unwrap();
    let rep = s.tensor_backward(out).unwrap();
    close(
        "grad",
        s.tensor_gradient(&rep, x).unwrap(),
        &[0.186324, 0.307196, 0.50648, 1.622757, 0.080792, 0.296451],
    );
}

/// 2nd-order: Hessian of logsumexp over a length-4 vector (full reduction).
#[test]
fn logsumexp_hessian_vector_matches_torch() {
    let mut s = sess();
    let x = s.tensor_variable(vec![0.5, 1.0, 1.5, 2.0], vec![4], true).unwrap();
    let y = s.tensor_logsumexp(x, 0).unwrap();
    let h = s.tensor_functional_hessian(y, x).unwrap();
    let diag: Vec<f64> = (0..4).map(|i| h[i * 4 + i]).collect();
    close("hess diag", &diag, &[0.091227, 0.139381, 0.199826, 0.24798]);
    // off-diagonal: H[0,1] = -p0*p1, H[0,2] = -p0*p2
    assert!((h[1] - (-0.016998)).abs() <= 1e-6, "H01 {}", h[1]);
    assert!((h[2] - (-0.028024)).abs() <= 1e-6, "H02 {}", h[2]);
}

/// 2nd-order over a NON-zero reduction dim ([2,3] along dim=1) — exercises the
/// cg backward's reshape+expand broadcast of grad_y back over the reduced axis.
#[test]
fn logsumexp_hessian_dim1_broadcast_matches_torch() {
    let mut s = sess();
    let x = s
        .tensor_variable(vec![0.5, 1.0, 1.5, 2.0, -1.0, 0.3], vec![2, 3], true)
        .unwrap();
    let y = s.tensor_logsumexp(x, 1).unwrap();
    let out = s.tensor_sum(y).unwrap();
    let h = s.tensor_functional_hessian(out, x).unwrap();
    let diag: Vec<f64> = (0..6).map(|i| h[i * 6 + i]).collect();
    close(
        "hess diag dim1",
        &diag,
        &[0.151607, 0.212827, 0.249958, 0.153044, 0.038764, 0.126255],
    );
    // same-row coupling negative; cross-row (idx 0 vs 3) is exactly 0.
    assert!((h[1] - (-0.057238)).abs() <= 1e-6, "H01 {}", h[1]);
    assert!(h[3].abs() <= 1e-12, "H03 should be 0, got {}", h[3]);
}
