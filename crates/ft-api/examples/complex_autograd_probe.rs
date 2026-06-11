//! Complex-tensor autograd parity probe vs torch (frankentorch-ng1hw).
//! Exercises the real<->complex bridge ops (complex / real / imag / abs /
//! view_as_real / view_as_complex) end to end: real leaves -> complex node ->
//! real loss -> backward, comparing the real-leaf gradients to torch.
//!   cargo run -q -p ft-api --example complex_autograd_probe
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn r(v: &[f64]) -> Vec<f64> {
    v.iter().map(|x| (x * 1e6).round() / 1e6).collect()
}

fn main() {
    // Chain A: loss = sum(real(z)*2 + imag(z)*3); grad re=2, im=3.
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let re = s.tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true).unwrap();
        let im = s
            .tensor_variable(vec![0.5, 1.5, 2.5, 3.5], vec![4], true)
            .unwrap();
        let z = s.tensor_complex(re, im).unwrap();
        let re_part = s.tensor_real(z).unwrap();
        let im_part = s.tensor_imag(z).unwrap();
        let a = s.tensor_mul_scalar(re_part, 2.0).unwrap();
        let b = s.tensor_mul_scalar(im_part, 3.0).unwrap();
        let sum = s.tensor_add(a, b).unwrap();
        let out = s.tensor_sum(sum).unwrap();
        let rep = s.tensor_backward(out).unwrap();
        println!("A re.grad {:?}  (torch [2,2,2,2])", r(s.tensor_gradient(&rep, re).unwrap()));
        println!("A im.grad {:?}  (torch [3,3,3,3])", r(s.tensor_gradient(&rep, im).unwrap()));
    }

    // Chain B: loss = sum(abs(z)); grad re = re/|z|, im = im/|z|.
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let re = s
            .tensor_variable(vec![3.0, 1.0, -2.0, 0.5], vec![4], true)
            .unwrap();
        let im = s
            .tensor_variable(vec![4.0, 1.0, 2.0, -0.5], vec![4], true)
            .unwrap();
        let z = s.tensor_complex(re, im).unwrap();
        let mag = s.tensor_abs(z).unwrap();
        let out = s.tensor_sum(mag).unwrap();
        let rep = s.tensor_backward(out).unwrap();
        println!(
            "B re.grad {:?}  (torch [0.6,0.707107,-0.707107,0.707107])",
            r(s.tensor_gradient(&rep, re).unwrap())
        );
        println!(
            "B im.grad {:?}  (torch [0.8,0.707107,0.707107,-0.707107])",
            r(s.tensor_gradient(&rep, im).unwrap())
        );
    }

    // Chain C: r[...,2] -> view_as_complex -> view_as_real -> *w -> sum; grad r = w.
    {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let rr = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], true)
            .unwrap();
        let z = s.tensor_view_as_complex(rr).unwrap();
        let r2 = s.tensor_view_as_real(z).unwrap();
        let w = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], false)
            .unwrap();
        let prod = s.tensor_mul(r2, w).unwrap();
        let out = s.tensor_sum(prod).unwrap();
        let rep = s.tensor_backward(out).unwrap();
        println!(
            "C r.grad {:?}  (torch [1,2,3,4,5,6])",
            r(s.tensor_gradient(&rep, rr).unwrap())
        );
    }
}
