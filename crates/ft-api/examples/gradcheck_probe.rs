//! Finite-difference gradient check for ops with subtle backward formulas
//! (var/std correction, logsumexp, log_softmax, softmax, prod, cumprod, norm).
//! Compares ft's analytic backward to central-difference numeric gradients.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

type Build = dyn Fn(&mut FrankenTorchSession, ft_autograd::TensorNodeId) -> ft_autograd::TensorNodeId;

fn loss_at(build: &Build, xv: &[f64]) -> f64 {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(xv.to_vec(), vec![xv.len()], false).unwrap();
    let loss = build(&mut s, x);
    s.tensor_values(loss).unwrap()[0]
}

fn analytic(build: &Build, xv: &[f64]) -> Vec<f64> {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(xv.to_vec(), vec![xv.len()], true).unwrap();
    let loss = build(&mut s, x);
    let report = s.tensor_backward(loss).unwrap();
    s.tensor_gradient(&report, x).unwrap().to_vec()
}

fn numeric(build: &Build, xv: &[f64]) -> Vec<f64> {
    let h = 1e-6;
    (0..xv.len())
        .map(|i| {
            let mut p = xv.to_vec();
            p[i] += h;
            let mut m = xv.to_vec();
            m[i] -= h;
            (loss_at(build, &p) - loss_at(build, &m)) / (2.0 * h)
        })
        .collect()
}

fn check(name: &str, build: &Build, xv: &[f64]) {
    let a = analytic(build, xv);
    let n = numeric(build, xv);
    let mut max_rel = 0.0f64;
    for (ga, gn) in a.iter().zip(n.iter()) {
        let rel = (ga - gn).abs() / (1.0 + gn.abs());
        max_rel = max_rel.max(rel);
    }
    let verdict = if max_rel < 1e-4 { "PASS" } else { "FAIL" };
    println!("{name:<16} max_rel={max_rel:.3e} {verdict}");
}

fn main() {
    let x = vec![0.5, -0.3, 1.2, 0.8, -1.0, 0.4];
    check("logsumexp", &|s, x| s.tensor_logsumexp(x, 0).unwrap(), &x);
    check("var_c1", &|s, x| s.tensor_var(x, 1).unwrap(), &x);
    check("std_c1", &|s, x| s.tensor_std(x, 1).unwrap(), &x);
    check("norm2", &|s, x| s.tensor_norm(x, 2.0).unwrap(), &x);
    check("prod", &|s, x| s.tensor_prod(x).unwrap(), &x);
    check(
        "cumprod_sum",
        &|s, x| {
            let c = s.tensor_cumprod(x, 0).unwrap();
            s.tensor_sum(c).unwrap()
        },
        &x,
    );
    check(
        "logsoftmax_sq",
        &|s, x| {
            let l = s.tensor_log_softmax(x, 0).unwrap();
            let sq = s.tensor_mul(l, l).unwrap();
            s.tensor_sum(sq).unwrap()
        },
        &x,
    );
    check(
        "softmax_sq",
        &|s, x| {
            let sm = s.tensor_softmax(x, 0).unwrap();
            let sq = s.tensor_mul(sm, sm).unwrap();
            s.tensor_sum(sq).unwrap()
        },
        &x,
    );
}
