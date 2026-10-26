use std::sync::Arc;

use ft_api::{FrankenTorchSession, IstftOptions, StftOptions};
use ft_core::{Complex128, DType, DenseTensor, Device, ExecutionMode, TensorMeta, TensorStorage};

fn strict_session() -> FrankenTorchSession {
    FrankenTorchSession::new(ExecutionMode::Strict)
}

fn assert_close(actual: &[f64], expected: &[f64]) {
    assert_eq!(actual.len(), expected.len());
    for (i, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-10,
            "element {i}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn stft_requires_grad_input_has_analytical_gradient() {
    let mut session = strict_session();
    let input = session
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
        .unwrap();
    let spectrum = session
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
    assert!(session.tensor_requires_grad(spectrum).unwrap());
    assert_eq!(session.tensor_dtype(spectrum).unwrap(), DType::Complex128);
    assert_eq!(session.tensor_shape(spectrum).unwrap(), vec![3, 1]);
    let packed = session.tensor_view_as_real(spectrum).unwrap();
    assert_close(
        &session.tensor_values(packed).unwrap(),
        &[10.0, 0.0, -2.0, 2.0, -2.0, 0.0],
    );
    let weights = session
        .tensor_variable(vec![1.0, 2.0, 3.0, 5.0, 7.0, 11.0], vec![3, 1, 2], false)
        .unwrap();
    let weighted = session.tensor_mul(packed, weights).unwrap();
    let loss = session.tensor_sum(weighted).unwrap();
    let report = session.tensor_backward(loss).unwrap();
    // For N=4 the real/imaginary DFT rows are [1,1,1,1],
    // [1,0,-1,0], [0,-1,0,1], and [1,-1,1,-1]. Their
    // weighted transpose gives a nonuniform signal cotangent.
    assert_close(
        session.tensor_gradient(&report, input).unwrap(),
        &[11.0, -11.0, 5.0, -1.0],
    );
}

#[test]
fn istft_requires_grad_input_has_analytical_gradient() {
    let mut session = strict_session();
    let tensor = DenseTensor::from_contiguous_values_complex128(
        vec![
            Complex128::new(10.0, 0.0),
            Complex128::new(-2.0, 2.0),
            Complex128::new(-2.0, 0.0),
        ],
        vec![3, 1],
        Device::Cpu,
    )
    .unwrap();
    let spectrogram = session.tensor_variable_from_storage(tensor, true);
    let output = session
        .tensor_istft(
            spectrogram,
            4,
            IstftOptions {
                hop_length: Some(4),
                win_length: Some(4),
                center: false,
                length: Some(4),
                ..IstftOptions::default()
            },
        )
        .unwrap();
    assert!(session.tensor_requires_grad(output).unwrap());
    assert_eq!(session.tensor_dtype(output).unwrap(), DType::F64);
    assert_eq!(session.tensor_shape(output).unwrap(), vec![4]);
    assert_close(
        &session.tensor_values(output).unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
    );
    let weights = session
        .tensor_variable(vec![1.0, 2.0, 4.0, 8.0], vec![4], false)
        .unwrap();
    let weighted = session.tensor_mul(output, weights).unwrap();
    let loss = session.tensor_sum(weighted).unwrap();
    let report = session.tensor_backward(loss).unwrap();
    // The real inverse has 1/N endpoint factors and 2/N interior
    // factors; DC and Nyquist imaginary components have zero derivative.
    assert_close(
        session.tensor_gradient(&report, spectrogram).unwrap(),
        &[3.75, 0.0, -1.5, 3.0, -1.25, 0.0],
    );
}

#[test]
fn stft_requires_grad_window_has_analytical_gradient() {
    let mut session = strict_session();
    let input = session
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], false)
        .unwrap();
    let window = session
        .tensor_variable(vec![1.0, 1.0, 1.0, 1.0], vec![4], true)
        .unwrap();

    let spectrum = session
        .tensor_stft(
            input,
            4,
            StftOptions {
                hop_length: Some(4),
                win_length: Some(4),
                window: Some(window),
                center: false,
                ..StftOptions::default()
            },
        )
        .unwrap();
    assert!(session.tensor_requires_grad(spectrum).unwrap());
    assert_eq!(session.tensor_dtype(spectrum).unwrap(), DType::Complex128);
    assert_eq!(session.tensor_shape(spectrum).unwrap(), vec![3, 1]);
    let packed = session.tensor_view_as_real(spectrum).unwrap();
    assert_close(
        &session.tensor_values(packed).unwrap(),
        &[10.0, 0.0, -2.0, 2.0, -2.0, 0.0],
    );
    let weights = session
        .tensor_variable(vec![1.0, 2.0, 3.0, 5.0, 7.0, 11.0], vec![3, 1, 2], false)
        .unwrap();
    let weighted = session.tensor_mul(packed, weights).unwrap();
    let loss = session.tensor_sum(weighted).unwrap();
    let report = session.tensor_backward(loss).unwrap();
    // Multiply the DFT adjoint [11,-11,5,-1] by the saved signal.
    assert_close(
        session.tensor_gradient(&report, window).unwrap(),
        &[11.0, -22.0, 15.0, -4.0],
    );
    assert!(session.tensor_gradient(&report, input).is_none());
}

#[test]
fn istft_requires_grad_window_has_analytical_gradient() {
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

    let output = session
        .tensor_istft(
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
        )
        .unwrap();
    assert!(session.tensor_requires_grad(output).unwrap());
    assert_eq!(session.tensor_dtype(output).unwrap(), DType::F64);
    assert_eq!(session.tensor_shape(output).unwrap(), vec![4]);
    assert_close(
        &session.tensor_values(output).unwrap(),
        &[1.0, 2.0, 3.0, 4.0],
    );
    let weights = session
        .tensor_variable(vec![1.0, 2.0, 4.0, 8.0], vec![4], false)
        .unwrap();
    let weighted = session.tensor_mul(output, weights).unwrap();
    let loss = session.tensor_sum(weighted).unwrap();
    let report = session.tensor_backward(loss).unwrap();
    // One nonoverlapping frame yields y=u*w/(w*w)=u/w.
    // At w=1 its window gradient is -u times the output cotangent.
    assert_close(
        session.tensor_gradient(&report, window).unwrap(),
        &[-1.0, -4.0, -12.0, -32.0],
    );
    assert!(session.tensor_gradient(&report, spectrogram).is_none());
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
    assert_eq!(
        session.tensor_gradient(&rep, input).unwrap(),
        &[1.0, 1.0, 1.0, 1.0]
    );
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
    assert_eq!(
        session.tensor_gradient(&rep, input).unwrap(),
        &[1.0, 1.0, 1.0, 1.0]
    );
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
