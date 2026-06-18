//! Parity probe for `AdaptiveLogSoftmaxWithLoss` vs torch (frankentorch-otfr9).
//! Prints ft's randomly-initialized weights + inputs + outputs in a parseable
//! `key|v0,v1,...` format; the companion check loads these into a
//! `torch.nn.AdaptiveLogSoftmaxWithLoss` with the SAME weights and compares
//! forward output/loss, log_prob, and predict.
//!   cargo run -q -p ft-nn --example adaptive_logsoftmax_probe
use ft_nn::{AdaptiveLogSoftmaxWithLoss, Module};
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn line(key: &str, v: &[f64]) {
    let s: Vec<String> = v.iter().map(|x| format!("{x:.17e}")).collect();
    println!("{key}|{}", s.join(","));
}

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let (n, in_f, classes) = (6usize, 8usize, 10usize);
    let cutoffs = [3usize, 6];
    let div_value = 2.0;
    let alswl =
        AdaptiveLogSoftmaxWithLoss::new(&mut s, in_f, classes, &cutoffs, div_value, true).unwrap();

    println!("config|{in_f},{classes},3,6,{div_value},1");
    // Head weight + bias.
    let hw = s.tensor_values(alswl.head().weight()).unwrap();
    line("head_weight", &hw);
    let hb = s.tensor_values(alswl.head().bias().unwrap()).unwrap();
    line("head_bias", &hb);
    // Tail cluster projections.
    for (i, (l0, l1)) in alswl.tail().iter().enumerate() {
        line(&format!("tail{i}_l0"), &s.tensor_values(l0.weight()).unwrap());
        line(&format!("tail{i}_l1"), &s.tensor_values(l1.weight()).unwrap());
    }

    let xdata: Vec<f64> = (0..n * in_f).map(|i| (i as f64 % 7.0) * 0.13 - 0.4).collect();
    line("input", &xdata);
    let tdata = vec![0.0, 2.0, 3.0, 5.0, 6.0, 9.0];
    line("target", &tdata);

    let x = s.tensor_variable(xdata, vec![n, in_f], true).unwrap();
    let target = s.tensor_variable(tdata, vec![n], false).unwrap();
    let (output, loss) = alswl.forward(&mut s, x, target).unwrap();
    line("ft_output", &s.tensor_values(output).unwrap());
    line("ft_loss", &s.tensor_values(loss).unwrap());
    let lp = alswl.log_prob(&mut s, x).unwrap();
    line("ft_log_prob", &s.tensor_values(lp).unwrap());
    let pred = alswl.predict(&mut s, x).unwrap();
    line("ft_predict", &s.tensor_values(pred).unwrap());
}
