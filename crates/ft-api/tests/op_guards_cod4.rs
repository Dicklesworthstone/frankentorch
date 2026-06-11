use std::fmt::Debug;
use std::sync::Arc;

use ft_api::{FrankenTorchSession, IstftOptions, StftOptions};
use ft_core::{Complex128, DType, DenseTensor, Device, ExecutionMode, TensorMeta, TensorStorage};

fn strict_session() -> FrankenTorchSession {
    FrankenTorchSession::new(ExecutionMode::Strict)
}

fn assert_fails_loud<T: Debug, E: Debug>(result: Result<T, E>, op: &str) {
    let err = result.expect_err(op);
    let msg = format!("{err:?}");
    assert!(
        msg.contains("autograd") || msg.contains("requires_grad"),
        "{op} should fail loud on unsupported autograd input, got {msg}"
    );
}

#[test]
fn stft_requires_grad_input_fails_loud() {
    let mut session = strict_session();
    let input = session
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
        .unwrap();

    assert_fails_loud(
        session.tensor_stft(
            input,
            4,
            StftOptions {
                hop_length: Some(4),
                win_length: Some(4),
                center: false,
                ..StftOptions::default()
            },
        ),
        "stft requires_grad input",
    );
}

#[test]
fn istft_requires_grad_input_fails_loud() {
    let mut session = strict_session();
    let spectrogram = session
        .tensor_variable(vec![1.0; 8], vec![2, 4], true)
        .unwrap();

    assert_fails_loud(
        session.tensor_istft(
            spectrogram,
            4,
            IstftOptions {
                hop_length: Some(4),
                win_length: Some(4),
                center: false,
                length: Some(4),
                ..IstftOptions::default()
            },
        ),
        "istft requires_grad input",
    );
}

#[test]
fn stft_requires_grad_window_fails_loud() {
    let mut session = strict_session();
    let input = session
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
        .unwrap();
    let window = session
        .tensor_variable(vec![1.0, 1.0, 1.0, 1.0], vec![4], true)
        .unwrap();

    assert_fails_loud(
        session.tensor_stft(
            input,
            4,
            StftOptions {
                hop_length: Some(4),
                win_length: Some(4),
                window: Some(window),
                center: false,
                ..StftOptions::default()
            },
        ),
        "stft requires_grad window",
    );
}

#[test]
fn istft_requires_grad_window_fails_loud() {
    let mut session = strict_session();
    let input = session
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
        .unwrap();
    let spectrogram = session
        .tensor_stft(
            input,
            4,
            StftOptions {
                hop_length: Some(4),
                win_length: Some(4),
                center: false,
                ..StftOptions::default()
            },
        )
        .unwrap();
    let window = session
        .tensor_variable(vec![1.0, 1.0, 1.0, 1.0], vec![4], true)
        .unwrap();

    assert_fails_loud(
        session.tensor_istft(
            spectrogram,
            4,
            IstftOptions {
                hop_length: Some(4),
                win_length: Some(4),
                window: Some(window),
                center: false,
                length: Some(4),
                ..IstftOptions::default()
            },
        ),
        "istft requires_grad window",
    );
}

#[test]
fn complex_requires_grad_parts_are_differentiable() {
    // frankentorch-ng1hw: tensor_complex is now differentiable. The complex
    // output's interleaved gradient de-interleaves back into the real/imag inputs.
    let mut session = strict_session();
    let real = session
        .tensor_variable(vec![1.0, 2.0], vec![2], true)
        .unwrap();
    let imag = session
        .tensor_variable(vec![3.0, 4.0], vec![2], true)
        .unwrap();
    let z = session.tensor_complex(real, imag).unwrap();
    assert_eq!(session.tensor_dtype(z).unwrap(), DType::Complex128);
    // loss = sum(imag(z)) -> grad real = 0, grad imag = 1.
    let im = session.tensor_imag(z).unwrap();
    let out = session.tensor_sum(im).unwrap();
    let rep = session.tensor_backward(out).unwrap();
    assert_eq!(session.tensor_gradient(&rep, real).unwrap(), &[0.0, 0.0]);
    assert_eq!(session.tensor_gradient(&rep, imag).unwrap(), &[1.0, 1.0]);
}

#[test]
fn view_as_complex_requires_grad_input_is_differentiable() {
    // frankentorch-ng1hw: view_as_complex is a differentiable layout bridge.
    let mut session = strict_session();
    let input = session
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
        .unwrap();
    let z = session.tensor_view_as_complex(input).unwrap();
    assert_eq!(session.tensor_dtype(z).unwrap(), DType::Complex128);
    let back = session.tensor_view_as_real(z).unwrap();
    let out = session.tensor_sum(back).unwrap();
    let rep = session.tensor_backward(out).unwrap();
    assert_eq!(session.tensor_gradient(&rep, input).unwrap(), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn view_as_real_requires_grad_complex_input_is_differentiable() {
    // frankentorch-ng1hw: view_as_real is a differentiable layout bridge; the
    // gradient is copied through unchanged (interleaved [re,im] layout).
    let mut session = strict_session();
    let tensor = DenseTensor::from_typed_storage(
        TensorMeta::from_shape(vec![2], DType::Complex128, Device::Cpu),
        TensorStorage::Complex128(Arc::new(vec![
            Complex128::new(1.0, 0.5),
            Complex128::new(2.0, -1.5),
        ])),
    )
    .unwrap();
    let input = session.tensor_variable_from_storage(tensor, true);
    let real_view = session.tensor_view_as_real(input).unwrap();
    let out = session.tensor_sum(real_view).unwrap();
    let rep = session.tensor_backward(out).unwrap();
    // d sum(view_as_real(z))/dz = ones over both re/im lanes (2 complex -> 4 reals).
    assert_eq!(session.tensor_gradient(&rep, input).unwrap(), &[1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn multi_dot_rejects_single_tensor_sequence() {
    let mut session = strict_session();
    let input = session
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false)
        .unwrap();

    let err = session
        .tensor_linalg_multi_dot(&[input])
        .expect_err("multi_dot requires two or more tensors");
    assert!(
        format!("{err:?}").contains("multi_dot: need at least two tensors"),
        "unexpected multi_dot single-input error: {err:?}"
    );
}
