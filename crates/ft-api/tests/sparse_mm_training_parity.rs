//! Sparse matrix training through the public API. Fixed products and adjoints
//! are independently evaluated matrix formulas; complex goldens were checked
//! against PyTorch 2.15.0 CPU. Finite differences and cubic cross derivatives
//! exercise the native forward and all three sparse adjoint contractions.

use std::sync::Arc;

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, BackwardOptions, TensorNodeId};
use ft_core::{
    Complex64, Complex128, DType, DenseI64Tensor, DenseTensor, Device, ExecutionMode, TensorMeta,
    TensorStorage,
};

fn indices(values: &[i64], shape: &[usize]) -> DenseI64Tensor {
    DenseI64Tensor::from_contiguous_values(values.to_vec(), shape.to_vec(), Device::Cpu).unwrap()
}

fn tensor(
    session: &mut FrankenTorchSession,
    values: &[f64],
    shape: &[usize],
    dtype: DType,
    requires_grad: bool,
) -> TensorNodeId {
    let tensor = DenseTensor::from_contiguous_values(values.to_vec(), shape.to_vec(), Device::Cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    session.tensor_variable_from_storage(tensor, requires_grad)
}

fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: got {actual}, expected {expected}"
        );
    }
}

fn weighted_loss(
    session: &mut FrankenTorchSession,
    output: TensorNodeId,
    weights: &[f64],
) -> TensorNodeId {
    let shape = session.tensor_shape(output).unwrap();
    let dtype = session.tensor_dtype(output).unwrap();
    let weights = tensor(session, weights, &shape, dtype, false);
    let weighted = session.tensor_mul(output, weights).unwrap();
    session.tensor_sum(weighted).unwrap()
}

#[test]
fn coo_duplicates_and_frozen_operands_keep_independent_value_gradients() {
    let index = indices(&[2, 0, 0, 2, 0, 1, 1, 2], &[2, 4]);
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
            for (values_grad, dense_grad) in [(true, true), (true, false), (false, true)] {
                for create_graph in [false, true] {
                    let mut session = FrankenTorchSession::new(mode);
                    let values = tensor(&mut session, &[2., 3., -1., 4.], &[4], dtype, values_grad);
                    let dense = tensor(
                        &mut session,
                        &[1., 2., 3., 5., 7., 11.],
                        &[3, 2],
                        dtype,
                        dense_grad,
                    );
                    let output = session
                        .tensor_sparse_coo_mm(&index, values, [3, 3], dense)
                        .unwrap();
                    assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
                    close(
                        &session.tensor_values_lossy_f64(output).unwrap(),
                        &[6., 10., 0., 0., 30., 48.],
                        0.,
                    );
                    let loss = weighted_loss(&mut session, output, &[1., -2., 3., 4., -1., 0.5]);
                    let mut options = BackwardOptions::for_mode(mode);
                    options.create_graph = create_graph;
                    let report = session.tensor_backward_with_options(loss, options).unwrap();
                    if values_grad {
                        close(report.gradient(values).unwrap(), &[0., -7., -7., -1.5], 0.);
                    } else {
                        assert!(report.gradient(values).is_none());
                    }
                    if dense_grad {
                        close(
                            report.gradient(dense).unwrap(),
                            &[-2., 1., 2., -4., -4., 2.],
                            0.,
                        );
                    } else {
                        assert!(report.gradient(dense).is_none());
                    }
                }
            }
        }
    }
}

#[test]
fn csr_unsorted_columns_and_empty_rows_train_both_operands() {
    let crow = indices(&[0, 2, 2, 3, 4], &[5]);
    let col = indices(&[1, 0, 2, 1], &[4]);
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
            for create_graph in [false, true] {
                let mut session = FrankenTorchSession::new(mode);
                let values = tensor(&mut session, &[3., 2., 4., -1.], &[4], dtype, true);
                let dense = tensor(
                    &mut session,
                    &[1., 2., 3., 5., 7., 11.],
                    &[3, 2],
                    dtype,
                    true,
                );
                let output = session
                    .tensor_sparse_csr_mm(&crow, &col, values, [4, 3], dense)
                    .unwrap();
                assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
                close(
                    &session.tensor_values_lossy_f64(output).unwrap(),
                    &[11., 19., 0., 0., 28., 44., -3., -5.],
                    0.,
                );
                let loss =
                    weighted_loss(&mut session, output, &[1., -2., 3., 4., -1., 0.5, 2., 3.]);
                let mut options = BackwardOptions::for_mode(mode);
                options.create_graph = create_graph;
                let report = session.tensor_backward_with_options(loss, options).unwrap();
                close(report.gradient(values).unwrap(), &[-7., -3., -1.5, 21.], 0.);
                close(
                    report.gradient(dense).unwrap(),
                    &[2., -4., 1., -9., -4., 2.],
                    0.,
                );
            }
        }
    }
}

#[test]
fn sparse_mm_accepts_logical_strides_and_routes_offset_view_gradients() {
    let index = indices(&[2, 0, 0, 2, 0, 1, 1, 2], &[2, 4]);
    for dtype in [DType::F32, DType::F64] {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let value_storage = DenseTensor::from_contiguous_values(
            vec![99., 2., 99., 3., 99., -1., 99., 4.],
            vec![8],
            Device::Cpu,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
        let value_meta =
            TensorMeta::from_shape_and_strides(vec![4], vec![2], 1, dtype, Device::Cpu).unwrap();
        let values = session.tensor_variable_from_storage(
            DenseTensor::from_typed_storage(value_meta, value_storage.typed_storage().clone())
                .unwrap(),
            true,
        );
        let dense_storage = DenseTensor::from_contiguous_values(
            vec![99., 1., 3., 7., 2., 5., 11.],
            vec![7],
            Device::Cpu,
        )
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
        let dense_meta =
            TensorMeta::from_shape_and_strides(vec![3, 2], vec![1, 3], 1, dtype, Device::Cpu)
                .unwrap();
        let dense = session.tensor_variable_from_storage(
            DenseTensor::from_typed_storage(dense_meta, dense_storage.typed_storage().clone())
                .unwrap(),
            true,
        );
        let output = session
            .tensor_sparse_coo_mm(&index, values, [3, 3], dense)
            .unwrap();
        close(
            &session.tensor_values_lossy_f64(output).unwrap(),
            &[6., 10., 0., 0., 30., 48.],
            0.,
        );
        let loss = weighted_loss(&mut session, output, &[1., -2., 3., 4., -1., 0.5]);
        let report = session.tensor_backward(loss).unwrap();
        close(report.gradient(values).unwrap(), &[0., -7., -7., -1.5], 0.);
        close(
            report.gradient(dense).unwrap(),
            &[-2., 1., 2., -4., -4., 2.],
            0.,
        );

        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let value_parent = tensor(
            &mut session,
            &[99., 2., 3., -1., 4., 99.],
            &[6],
            dtype,
            true,
        );
        let dense_parent = tensor(
            &mut session,
            &[99., 99., 1., 2., 3., 5., 7., 11.],
            &[4, 2],
            dtype,
            true,
        );
        let values = session.tensor_narrow(value_parent, 0, 1, 4).unwrap();
        let dense = session.tensor_narrow(dense_parent, 0, 1, 3).unwrap();
        let output = session
            .tensor_sparse_coo_mm(&index, values, [3, 3], dense)
            .unwrap();
        let loss = weighted_loss(&mut session, output, &[1., -2., 3., 4., -1., 0.5]);
        let report = session.tensor_backward(loss).unwrap();
        close(
            report.gradient(value_parent).unwrap(),
            &[0., 0., -7., -7., -1.5, 0.],
            0.,
        );
        close(
            report.gradient(dense_parent).unwrap(),
            &[0., 0., -2., 1., 2., -4., -4., 2.],
            0.,
        );
    }
}

#[test]
fn sparse_bilinear_adjoints_support_mixed_second_and_third_derivatives() {
    let index = indices(&[0, 0], &[2, 1]);
    let crow = indices(&[0, 1], &[2]);
    let col = indices(&[0], &[1]);
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
            for csr in [false, true] {
                let mut session = FrankenTorchSession::new(mode);
                let value = tensor(&mut session, &[2.], &[1], dtype, true);
                let dense = tensor(&mut session, &[3.], &[1, 1], dtype, true);
                let output = if csr {
                    session.tensor_sparse_csr_mm(&crow, &col, value, [1, 1], dense)
                } else {
                    session.tensor_sparse_coo_mm(&index, value, [1, 1], dense)
                }
                .unwrap();
                let square = session.tensor_mul(output, output).unwrap();
                let cube = session.tensor_mul(square, output).unwrap();
                let loss = session.tensor_sum(cube).unwrap();
                let mut options = BackwardOptions::for_mode(mode);
                options.create_graph = true;
                options.retain_graph = true;
                let first = session.tensor_backward_with_options(loss, options).unwrap();
                close(first.gradient(value).unwrap(), &[324.], 1e-10);
                close(first.gradient(dense).unwrap(), &[216.], 1e-10);
                let first_value = session
                    .tensor_sum(first.gradient_node(value).unwrap())
                    .unwrap();
                let second = session
                    .tensor_backward_with_options(first_value, options)
                    .unwrap();
                close(second.gradient(value).unwrap(), &[324.], 1e-10);
                close(second.gradient(dense).unwrap(), &[324.], 1e-10);
                let second_value = session
                    .tensor_sum(second.gradient_node(value).unwrap())
                    .unwrap();
                let third = session.tensor_backward(second_value).unwrap();
                close(third.gradient(value).unwrap(), &[162.], 1e-10);
                close(third.gradient(dense).unwrap(), &[324.], 1e-10);
            }
        }
    }
}

fn independent_dense_loss(parameters: &[f64]) -> f64 {
    // Explicitly assemble the mathematical dense matrix, including duplicate
    // summation, then use a separate ordinary dense matrix product.
    let matrix = [
        0.,
        parameters[1] + parameters[2],
        0.,
        0.,
        0.,
        0.,
        parameters[0],
        0.,
        parameters[3],
    ];
    let dense = &parameters[4..];
    let mut loss = 0.;
    for row in 0..3 {
        for column in 0..2 {
            let output: f64 = (0..3)
                .map(|inner| matrix[row * 3 + inner] * dense[inner * 2 + column])
                .sum();
            loss += output.sin() + 0.1 * output * output;
        }
    }
    loss
}

#[test]
fn sparse_mm_nonlinear_gradients_match_independent_finite_differences() {
    let index = indices(&[2, 0, 0, 2, 0, 1, 1, 2], &[2, 4]);
    let parameters = [0.7, -1.2, 0.4, 1.3, 0.5, -0.2, 0.9, 1.1, -0.8, 0.6];
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        let mut session = FrankenTorchSession::new(mode);
        let values = tensor(&mut session, &parameters[..4], &[4], DType::F64, true);
        let dense = tensor(&mut session, &parameters[4..], &[3, 2], DType::F64, true);
        let output = session
            .tensor_sparse_coo_mm(&index, values, [3, 3], dense)
            .unwrap();
        let sin = session.tensor_sin(output).unwrap();
        let square = session.tensor_mul(output, output).unwrap();
        let scaled = session.tensor_mul_scalar(square, 0.1).unwrap();
        let terms = session.tensor_add(sin, scaled).unwrap();
        let loss = session.tensor_sum(terms).unwrap();
        close(
            &session.tensor_values(loss).unwrap(),
            &[independent_dense_loss(&parameters)],
            1e-12,
        );
        let report = session.tensor_backward(loss).unwrap();
        let actual = [
            report.gradient(values).unwrap(),
            report.gradient(dense).unwrap(),
        ]
        .concat();
        let step = 1e-6;
        for slot in 0..parameters.len() {
            let mut plus = parameters;
            let mut minus = parameters;
            plus[slot] += step;
            minus[slot] -= step;
            let expected =
                (independent_dense_loss(&plus) - independent_dense_loss(&minus)) / (2. * step);
            close(&[actual[slot]], &[expected], 1e-8);
        }
    }
}

#[test]
fn empty_sparse_products_preserve_zero_gradients_without_reading_absent_entries() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for create_graph in [false, true] {
                for (rows, columns, entries) in [(3, 2, 0), (0, 2, 0), (3, 0, 1)] {
                    let mut session = FrankenTorchSession::new(mode);
                    let index = if entries == 0 {
                        indices(&[], &[2, 0])
                    } else {
                        indices(&[1, 0], &[2, 1])
                    };
                    let values = tensor(&mut session, &vec![2.; entries], &[entries], dtype, true);
                    let dense = tensor(
                        &mut session,
                        &vec![f64::NAN; 2 * columns],
                        &[2, columns],
                        dtype,
                        true,
                    );
                    let output = session
                        .tensor_sparse_coo_mm(&index, values, [rows, 2], dense)
                        .unwrap();
                    assert_eq!(session.tensor_shape(output).unwrap(), [rows, columns]);
                    close(
                        &session.tensor_values_lossy_f64(output).unwrap(),
                        &vec![0.; rows * columns],
                        0.,
                    );
                    let loss = session.tensor_sum(output).unwrap();
                    let mut options = BackwardOptions::for_mode(mode);
                    options.create_graph = create_graph;
                    let report = session.tensor_backward_with_options(loss, options).unwrap();
                    close(report.gradient(values).unwrap(), &vec![0.; entries], 0.);
                    close(report.gradient(dense).unwrap(), &vec![0.; 2 * columns], 0.);
                }
            }
        }
    }
}

#[test]
fn sparse_mm_no_grad_uses_one_native_leaf_and_preserves_dtype() {
    let index = indices(&[0, 0], &[2, 1]);
    for dtype in [DType::F32, DType::F64, DType::F16, DType::BF16] {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let values = tensor(&mut session, &[2.], &[1], dtype, true);
        let dense = tensor(&mut session, &[3., 4.], &[1, 2], dtype, true);
        let count = session.autograd_graph_node_count();
        let output = session
            .with_no_grad(|session| session.tensor_sparse_coo_mm(&index, values, [1, 1], dense))
            .unwrap();
        assert_eq!(session.autograd_graph_node_count(), count + 1);
        assert!(!session.tensor_requires_grad(output).unwrap());
        assert!(session.tensor_grad_fn(output).unwrap().is_none());
        assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
        close(
            &session.tensor_values_lossy_f64(output).unwrap(),
            &[6., 8.],
            0.,
        );
        assert!(session.is_grad_enabled());
    }
}

#[test]
fn invalid_sparse_requests_fail_before_recording_nodes() {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let values = tensor(&mut session, &[2.], &[1], DType::F32, true);
    let dense = tensor(&mut session, &[3., 4.], &[1, 2], DType::F32, true);
    for index in [
        indices(&[-1, 0], &[2, 1]),
        indices(&[1, 0], &[2, 1]),
        indices(&[0, 1], &[2, 1]),
        indices(&[0], &[1, 1]),
        indices(&[], &[2, 0]),
    ] {
        let count = session.autograd_graph_node_count();
        assert!(
            session
                .tensor_sparse_coo_mm(&index, values, [1, 1], dense)
                .is_err()
        );
        assert_eq!(session.autograd_graph_node_count(), count);
        assert!(session.is_grad_enabled());
    }
    let index = indices(&[0, 0], &[2, 1]);
    let wrong_dtype = tensor(&mut session, &[3., 4.], &[1, 2], DType::F64, false);
    let wrong_shape = tensor(&mut session, &[3., 4.], &[2, 1], DType::F32, false);
    for rhs in [wrong_dtype, wrong_shape, TensorNodeId(usize::MAX)] {
        let count = session.autograd_graph_node_count();
        assert!(
            session
                .tensor_sparse_coo_mm(&index, values, [1, 1], rhs)
                .is_err()
        );
        assert_eq!(session.autograd_graph_node_count(), count);
    }
    let duplicate_values = tensor(&mut session, &[1., 2.], &[2], DType::F32, true);
    let count = session.autograd_graph_node_count();
    assert!(
        session
            .tensor_sparse_csr_mm(
                &indices(&[0, 2], &[2]),
                &indices(&[0, 0], &[2]),
                duplicate_values,
                [1, 1],
                dense
            )
            .is_err()
    );
    assert_eq!(session.autograd_graph_node_count(), count);
}

#[test]
fn sparse_adjoints_reject_mutated_saved_frozen_operands() {
    let index = indices(&[0, 0], &[2, 1]);
    for frozen_values in [false, true] {
        for create_graph in [false, true] {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let values = tensor(&mut session, &[2.], &[1], DType::F64, !frozen_values);
            let dense = tensor(&mut session, &[3., 4.], &[1, 2], DType::F64, frozen_values);
            let output = session
                .tensor_sparse_coo_mm(&index, values, [1, 1], dense)
                .unwrap();
            let loss = session.tensor_sum(output).unwrap();
            let frozen = if frozen_values { values } else { dense };
            session.tensor_fill_(frozen, 9.).unwrap();
            let mut options = BackwardOptions::for_mode(ExecutionMode::Strict);
            options.create_graph = create_graph;
            assert!(session.tensor_backward_with_options(loss, options).is_err());
        }
    }
}

#[test]
fn sparse_adjoints_guard_original_strided_operands_after_state_reload() {
    let coo = indices(&[0, 1, 1, 0], &[2, 2]);
    let crow = indices(&[0, 1, 2], &[3]);
    let columns = indices(&[1, 0], &[2]);
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for frozen_values in [false, true] {
                for create_graph in [false, true] {
                    for use_csr in [false, true] {
                        let mut session = FrankenTorchSession::new(mode);
                        let value_storage = DenseTensor::from_contiguous_values(
                            vec![99., 2., 99., 3.],
                            vec![4],
                            Device::Cpu,
                        )
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap();
                        let values = DenseTensor::from_typed_storage(
                            TensorMeta::from_shape_and_strides(
                                vec![2],
                                vec![2],
                                1,
                                dtype,
                                Device::Cpu,
                            )
                            .unwrap(),
                            value_storage.typed_storage().clone(),
                        )
                        .unwrap();
                        let dense_storage = DenseTensor::from_contiguous_values(
                            vec![99., 1., 4., 99., 2., 5.],
                            vec![6],
                            Device::Cpu,
                        )
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap();
                        let dense = DenseTensor::from_typed_storage(
                            TensorMeta::from_shape_and_strides(
                                vec![2, 2],
                                vec![1, 3],
                                1,
                                dtype,
                                Device::Cpu,
                            )
                            .unwrap(),
                            dense_storage.typed_storage().clone(),
                        )
                        .unwrap();
                        let values = session.tensor_variable_from_storage(values, !frozen_values);
                        let dense = session.tensor_variable_from_storage(dense, frozen_values);
                        let output = if use_csr {
                            session.tensor_sparse_csr_mm(&crow, &columns, values, [2, 2], dense)
                        } else {
                            session.tensor_sparse_coo_mm(&coo, values, [2, 2], dense)
                        }
                        .unwrap();
                        let loss = session.tensor_sum(output).unwrap();
                        let frozen = if frozen_values { values } else { dense };
                        let original = session.tensor_snapshot(frozen).unwrap();
                        assert!(!original.meta().is_contiguous());
                        let replacement = DenseTensor::from_contiguous_values(
                            vec![9.; original.meta().numel()],
                            original.meta().shape().to_vec(),
                            Device::Cpu,
                        )
                        .unwrap()
                        .to_dtype(dtype)
                        .unwrap();
                        session.tensor_load_state(&[(frozen, replacement)]).unwrap();
                        assert_eq!(
                            session.tensor_snapshot(frozen).unwrap().meta(),
                            original.meta()
                        );
                        let mut options = BackwardOptions::for_mode(mode);
                        options.create_graph = create_graph;
                        let error = session
                            .tensor_backward_with_options(loss, options)
                            .unwrap_err();
                        assert!(
                            matches!(
                                error,
                                AutogradError::BorrowedInputVersionMismatch {
                                    node,
                                    recorded: 0,
                                    observed: 1,
                                } if node == frozen
                            ),
                            "{dtype:?}, {mode:?}, CSR={use_csr}, frozen_values={frozen_values}, create_graph={create_graph}: {error:?}"
                        );
                    }
                }
            }
        }
    }
}

fn complex_tensor(
    session: &mut FrankenTorchSession,
    values: &[(f64, f64)],
    shape: &[usize],
    dtype: DType,
    requires_grad: bool,
) -> TensorNodeId {
    let storage = if dtype == DType::Complex64 {
        TensorStorage::Complex64(Arc::new(
            values
                .iter()
                .map(|&(re, im)| Complex64::new(re as f32, im as f32))
                .collect(),
        ))
    } else {
        TensorStorage::Complex128(Arc::new(
            values
                .iter()
                .map(|&(re, im)| Complex128::new(re, im))
                .collect(),
        ))
    };
    let tensor = DenseTensor::from_typed_storage(
        TensorMeta::from_shape(shape.to_vec(), dtype, Device::Cpu),
        storage,
    )
    .unwrap();
    session.tensor_variable_from_storage(tensor, requires_grad)
}

#[test]
fn complex_csr_unsorted_columns_and_empty_rows_preserve_conjugate_adjoints() {
    let crow = indices(&[0, 2, 2, 3, 4], &[5]);
    let columns = indices(&[1, 0, 2, 1], &[4]);
    // Independent dense-matrix products and adjoints. Row 0 stores column 1
    // before column 0; row 1 is empty and has nonzero loss weights. The VJP is
    // G * conj(B) for stored values and conj(A)^T * G for the dense operand.
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::Complex64, DType::Complex128] {
            for (value_grad, dense_grad) in [(true, true), (true, false), (false, true)] {
                let mut session = FrankenTorchSession::new(mode);
                let values = complex_tensor(
                    &mut session,
                    &[(3., -1.), (2., 0.5), (4., -2.), (-1., 0.25)],
                    &[4],
                    dtype,
                    value_grad,
                );
                let dense = complex_tensor(
                    &mut session,
                    &[
                        (1., 1.),
                        (2., -1.),
                        (3., 0.5),
                        (5., 2.),
                        (7., -1.),
                        (11., 0.25),
                    ],
                    &[3, 2],
                    dtype,
                    dense_grad,
                );
                let output = session
                    .tensor_sparse_csr_mm(&crow, &columns, values, [4, 3], dense)
                    .unwrap();
                assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
                assert_eq!(session.tensor_shape(output).unwrap(), [4, 2]);
                let real = session.tensor_real(output).unwrap();
                let imag = session.tensor_imag(output).unwrap();
                close(
                    &session.tensor_values_lossy_f64(real).unwrap(),
                    &[11., 21.5, 0., 0., 26., 44.5, -3.125, -5.5],
                    0.,
                );
                close(
                    &session.tensor_values_lossy_f64(imag).unwrap(),
                    &[1., 0., 0., 0., -18., -21., 0.25, -0.75],
                    0.,
                );
                let real_loss =
                    weighted_loss(&mut session, real, &[1., -2., 3., 4., -1., 0.5, 2., 3.]);
                let imag_loss =
                    weighted_loss(&mut session, imag, &[2., 1., -1., 3., 2., -0.5, -0.25, 1.5]);
                let loss = session.tensor_add(real_loss, imag_loss).unwrap();
                close(
                    &session.tensor_values_lossy_f64(loss).unwrap(),
                    &[-83.1875],
                    0.,
                );
                let report = session.tensor_backward(loss).unwrap();
                if value_grad {
                    close(
                        report.gradient(values).unwrap(),
                        &[-4., 14.5, -2., 1., -3.625, 7.375, 23.875, -0.25],
                        0.,
                    );
                } else {
                    assert!(report.gradient(values).is_none());
                }
                if dense_grad {
                    close(
                        report.gradient(dense).unwrap(),
                        &[
                            3., 3.5, -3.5, 3., -1.0625, 6.75, -9.625, -1.25, -8., 6., 3., -1.,
                        ],
                        0.,
                    );
                } else {
                    assert!(report.gradient(dense).is_none());
                }
            }
        }
    }
}

#[test]
fn complex_sparse_mm_conjugate_adjoints_match_pytorch() {
    let index = indices(&[2, 0, 0, 2, 0, 1, 1, 2], &[2, 4]);
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::Complex64, DType::Complex128] {
            for (value_grad, dense_grad) in [(true, true), (true, false), (false, true)] {
                let mut session = FrankenTorchSession::new(mode);
                let values = complex_tensor(
                    &mut session,
                    &[(2., 0.5), (3., -1.), (-1., 0.25), (4., -2.)],
                    &[4],
                    dtype,
                    value_grad,
                );
                let dense = complex_tensor(
                    &mut session,
                    &[
                        (1., 1.),
                        (2., -1.),
                        (3., 0.5),
                        (5., 2.),
                        (7., -1.),
                        (11., 0.25),
                    ],
                    &[3, 2],
                    dtype,
                    dense_grad,
                );
                let output = session
                    .tensor_sparse_coo_mm(&index, values, [3, 3], dense)
                    .unwrap();
                assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
                let real = session.tensor_real(output).unwrap();
                let imag = session.tensor_imag(output).unwrap();
                close(
                    &session.tensor_values_lossy_f64(real).unwrap(),
                    &[6.375, 11.5, 0., 0., 27.5, 49.],
                    0.,
                );
                close(
                    &session.tensor_values_lossy_f64(imag).unwrap(),
                    &[-1.25, 0.25, 0., 0., -15.5, -22.],
                    0.,
                );
                let real_loss = weighted_loss(&mut session, real, &[1., -2., 3., 4., -1., 0.5]);
                let imag_loss = weighted_loss(&mut session, imag, &[2., 1., -1., 3., 2., -0.5]);
                let loss = session.tensor_add(real_loss, imag_loss).unwrap();
                let report = session.tensor_backward(loss).unwrap();
                if value_grad {
                    close(
                        report.gradient(values).unwrap(),
                        &[2.5, 2.5, -4., 14.5, -4., 14.5, -3.625, 7.375],
                        0.,
                    );
                } else {
                    assert!(report.gradient(values).is_none());
                }
                if dense_grad {
                    close(
                        report.gradient(dense).unwrap(),
                        &[
                            -1., 4.5, 0.75, -1.25, 0.5, 4.75, -4.75, 0.5, -8., 6., 3., -1.,
                        ],
                        0.,
                    );
                } else {
                    assert!(report.gradient(dense).is_none());
                }
            }
        }
    }
}
