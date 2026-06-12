//! FFT forward parity probe vs torch 2.12. Prints ft re/im; compare to oracle.
//!   cargo run -q -p ft-api --example fft_probe
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let data = vec![1.0, 2.0, 3.0, 4.0, 0.0, -1.0, 2.0, 1.0];
    let reim = |s: &mut FrankenTorchSession, c| {
        let re = s.tensor_real(c).unwrap();
        let rev = s.tensor_values(re).unwrap();
        let im = s.tensor_imag(c).unwrap();
        let imv = s.tensor_values(im).unwrap();
        let r = |v: &[f64]| v.iter().map(|x| (x * 1e5).round() / 1e5).collect::<Vec<_>>();
        (r(&rev), r(&imv))
    };

    let x = s.tensor_variable(data.clone(), vec![8], false).unwrap();
    let f = s.tensor_fft(x, None).unwrap();
    let (re, im) = reim(&mut s, f);
    println!("fft_re {re:?}");
    println!("fft_im {im:?}");

    let x = s.tensor_variable(data.clone(), vec![8], false).unwrap();
    let rf = s.tensor_rfft(x, None).unwrap();
    let (re, im) = reim(&mut s, rf);
    println!("rfft_re {re:?}");
    println!("rfft_im {im:?}");

    let x = s.tensor_variable(data.clone(), vec![8], false).unwrap();
    let fo = s.tensor_fft_norm(x, None, "ortho").unwrap();
    let (re, _) = reim(&mut s, fo);
    println!("fft_ortho_re {:?}", &re[..3]);

    let x = s.tensor_variable(data.clone(), vec![8], false).unwrap();
    let f2 = s.tensor_fft(x, None).unwrap();
    let inv = s.tensor_ifft(f2, None).unwrap();
    let (re, _) = reim(&mut s, inv);
    println!("ifft_re {:?}", &re[..4]);
}
