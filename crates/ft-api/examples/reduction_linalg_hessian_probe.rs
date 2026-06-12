//! Double-backward (Hessian) parity probe for REDUCTION + LINALG scalar losses
//! vs torch. Fresh axis: the elementwise 2nd-derivative surface is mature, but
//! reduction/dim-op and linalg backward-of-backward is unprobed.
//!   cargo run -q -p ft-api --example reduction_linalg_hessian_probe
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn r6(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| (x * 1e6).round() / 1e6).collect()
}

fn main() {
    let x4 = vec![0.5f64, 1.0, 1.5, 2.0];

    // Reduction losses on a length-4 vector. Print full 4x4 Hessian diagonal +
    // a couple off-diagonals (Hessians here are dense, so off-diag matters).
    macro_rules! red {
        ($name:literal, $build:expr) => {{
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s.tensor_variable(x4.clone(), vec![4], true).unwrap();
            let build: &dyn Fn(
                &mut FrankenTorchSession,
                ft_autograd::TensorNodeId,
            ) -> Result<ft_autograd::TensorNodeId, ft_autograd::AutogradError> = &$build;
            match build(&mut s, x).and_then(|y| s.tensor_functional_hessian(y, x)) {
                Ok(h) => {
                    // print diagonal + H[0,1] + H[0,2]
                    let diag: Vec<f64> = (0..4).map(|i| h[i * 4 + i]).collect();
                    println!("{:<14} diag={:?} H01={:.6} H02={:.6}", $name, r6(&diag), h[1], h[2]);
                }
                Err(e) => println!("{:<14} ERR {:?}", $name, e),
            }
        }};
    }

    red!("logsumexp", |s: &mut FrankenTorchSession, x| s.tensor_logsumexp(x, 0));
    red!("softmax_sq", |s: &mut FrankenTorchSession, x| {
        let sm = s.tensor_softmax(x, 0)?;
        let sq = s.tensor_mul(sm, sm)?;
        s.tensor_sum(sq)
    });
    red!("logsoftmax_w", |s: &mut FrankenTorchSession, x| {
        let ls = s.tensor_log_softmax(x, 0)?;
        let w = s.tensor_variable(vec![0.1, 0.2, 0.3, 0.4], vec![4], false)?;
        let prod = s.tensor_mul(ls, w)?;
        s.tensor_sum(prod)
    });
    red!("var_unbiased", |s: &mut FrankenTorchSession, x| s.tensor_var(x, 1));
    red!("std_unbiased", |s: &mut FrankenTorchSession, x| s.tensor_std(x, 1));
    red!("mean_sq", |s: &mut FrankenTorchSession, x| {
        let sq = s.tensor_mul(x, x)?;
        s.tensor_mean(sq)
    });
    red!("norm2", |s: &mut FrankenTorchSession, x| s.tensor_norm(x, 2.0));

    // Linalg losses on a 2x2 SPD matrix A = [[2,0.5],[0.5,1.5]]. Hessian is 4x4.
    let a = vec![2.0f64, 0.5, 0.5, 1.5];
    macro_rules! lin {
        ($name:literal, $build:expr) => {{
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let m = s.tensor_variable(a.clone(), vec![2, 2], true).unwrap();
            let build: &dyn Fn(
                &mut FrankenTorchSession,
                ft_autograd::TensorNodeId,
            ) -> Result<ft_autograd::TensorNodeId, ft_autograd::AutogradError> = &$build;
            match build(&mut s, m).and_then(|y| s.tensor_functional_hessian(y, m)) {
                Ok(h) => {
                    let diag: Vec<f64> = (0..4).map(|i| h[i * 4 + i]).collect();
                    println!("{:<14} diag={:?} H03={:.6}", $name, r6(&diag), h[3]);
                }
                Err(e) => println!("{:<14} ERR {:?}", $name, e),
            }
        }};
    }

    lin!("logdet", |s: &mut FrankenTorchSession, m| s.tensor_logdet(m));
    lin!("sum_inv", |s: &mut FrankenTorchSession, m| {
        let inv = s.tensor_inverse(m)?;
        s.tensor_sum(inv)
    });
    lin!("trace_inv", |s: &mut FrankenTorchSession, m| {
        let inv = s.tensor_inverse(m)?;
        s.tensor_trace(inv)
    });
}
