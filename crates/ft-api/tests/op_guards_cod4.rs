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
fn complex_requires_grad_parts_fail_loud() {
    let mut session = strict_session();
    let real_grad = session.tensor_variable(vec![1.0], vec![1], true).unwrap();
    let imag_no_grad = session.tensor_variable(vec![2.0], vec![1], false).unwrap();

    assert_fails_loud(
        session.tensor_complex(real_grad, imag_no_grad),
        "complex requires_grad real",
    );

    let real_no_grad = session.tensor_variable(vec![1.0], vec![1], false).unwrap();
    let imag_grad = session.tensor_variable(vec![2.0], vec![1], true).unwrap();
    assert_fails_loud(
        session.tensor_complex(real_no_grad, imag_grad),
        "complex requires_grad imag",
    );
}

#[test]
fn view_as_complex_requires_grad_input_fails_loud() {
    let mut session = strict_session();
    let input = session
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true)
        .unwrap();

    assert_fails_loud(
        session.tensor_view_as_complex(input),
        "view_as_complex requires_grad input",
    );
}

#[test]
fn view_as_real_requires_grad_complex_input_fails_loud() {
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

    assert_fails_loud(
        session.tensor_view_as_real(input),
        "view_as_real requires_grad complex input",
    );
}
