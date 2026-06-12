//! Double-backward (Hessian) probe for solve / cholesky / cholesky_solve vs torch.
//!   cargo run -q -p ft-api --example solve_chol_hessian_probe
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn r6(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| (x * 1e6).round() / 1e6).collect()
}

fn main() {
    // A = [[2,0.5],[0.5,1.5]] SPD; b = [1, 2].
    let a = vec![2.0f64, 0.5, 0.5, 1.5];
    let b = vec![1.0f64, 2.0];

    // solve(A,b) — loss = sum(solve(A,b)); Hessian over A (4x4).
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let m = s.tensor_variable(a.clone(), vec![2, 2], true).unwrap();
        let bv = s.tensor_variable(b.clone(), vec![2], false).unwrap();
        match s
            .tensor_linalg_solve(m, bv)
            .and_then(|x| s.tensor_sum(x))
            .and_then(|y| s.tensor_functional_hessian(y, m))
        {
            Ok(h) => {
                let diag: Vec<f64> = (0..4).map(|i| h[i * 4 + i]).collect();
                println!("solve_A     diag={:?} H03={:.6}", r6(&diag), h[3]);
            }
            Err(e) => println!("solve_A     ERR {e:?}"),
        }
    }

    // cholesky(A) — loss = sum(L); Hessian over A.
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let m = s.tensor_variable(a.clone(), vec![2, 2], true).unwrap();
        match s
            .tensor_cholesky(m, false)
            .and_then(|l| s.tensor_sum(l))
            .and_then(|y| s.tensor_functional_hessian(y, m))
        {
            Ok(h) => {
                let diag: Vec<f64> = (0..4).map(|i| h[i * 4 + i]).collect();
                println!("chol_A      diag={:?} H00={:.6}", r6(&diag), h[0]);
            }
            Err(e) => println!("chol_A      ERR {e:?}"),
        }
    }

    // cholesky_solve(b, L) — loss = sum(.); Hessian over b.
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let m = s.tensor_variable(a.clone(), vec![2, 2], false).unwrap();
        let l = s.tensor_cholesky(m, false).unwrap();
        let bv = s.tensor_variable(b.clone(), vec![2, 1], true).unwrap();
        match s
            .tensor_cholesky_solve(bv, l, false)
            .and_then(|x| s.tensor_sum(x))
            .and_then(|y| s.tensor_functional_hessian(y, bv))
        {
            Ok(h) => println!("cholsolve_b diag={:?}", r6(&h)),
            Err(e) => println!("cholsolve_b ERR {e:?}"),
        }
    }
}
