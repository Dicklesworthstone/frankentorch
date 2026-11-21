//! Scatter extrema and mean training regressions. The fixed goldens follow
//! PyTorch's scatter_reduce(include_self=true) and its explicit tie-count
//! adjoint in torch/csrc/autograd/FunctionsManual.cpp. Cubic losses exercise
//! higher derivatives through the incoming cotangent, including tied winners.

use ft_api::FrankenTorchSession;
use ft_autograd::{BackwardOptions, TensorNodeId};
use ft_core::{DType, ExecutionMode};

fn tensor(
    session: &mut FrankenTorchSession,
    values: &[f64],
    shape: &[usize],
    dtype: DType,
    requires_grad: bool,
) -> TensorNodeId {
    if dtype == DType::F32 {
        session
            .tensor_variable_f32(
                values.iter().map(|&value| value as f32).collect(),
                shape.to_vec(),
                requires_grad,
            )
            .unwrap()
    } else {
        session
            .tensor_variable(values.to_vec(), shape.to_vec(), requires_grad)
            .unwrap()
    }
}

fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "element {index}: {actual} must be NaN");
        } else if expected.is_infinite() {
            assert_eq!(actual, expected, "element {index}");
        } else {
            assert!(
                (actual - expected).abs() <= tolerance,
                "element {index}: got {actual}, expected {expected}"
            );
        }
    }
}

fn weighted_loss(
    session: &mut FrankenTorchSession,
    output: TensorNodeId,
    weights: &[f64],
) -> TensorNodeId {
    let dtype = session.tensor_dtype(output).unwrap();
    let shape = session.tensor_shape(output).unwrap();
    let weights = tensor(session, weights, &shape, dtype, false);
    let weighted = session.tensor_mul(output, weights).unwrap();
    session.tensor_sum(weighted).unwrap()
}

#[test]
fn extrema_share_ties_across_input_and_repeated_sources() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for (input_grad, src_grad) in [(true, true), (true, false), (false, true)] {
                for reduce in ["amax", "amin"] {
                    let mut session = FrankenTorchSession::new(mode);
                    let input = tensor(
                        &mut session,
                        &[2.0, 5.0, -3.0, 7.0],
                        &[4],
                        dtype,
                        input_grad,
                    );
                    let src = tensor(
                        &mut session,
                        &[2.0, 2.0, 1.0, 5.0, -4.0, -3.0, 9.0, 9.0],
                        &[8],
                        dtype,
                        src_grad,
                    );
                    let index = tensor(
                        &mut session,
                        &[0.0, 0.0, 0.0, 1.0, 2.0, 2.0, 3.0, 3.0],
                        &[8],
                        DType::F64,
                        false,
                    );
                    let output = session
                        .tensor_scatter_reduce(input, 0, index, src, reduce)
                        .unwrap();
                    assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
                    let (expected, grad_input, grad_src) = if reduce == "amax" {
                        (
                            [2.0, 5.0, -3.0, 9.0],
                            [1.0, -2.0, 2.5, 0.0],
                            [1.0, 1.0, 0.0, -2.0, 0.0, 2.5, 3.0, 3.0],
                        )
                    } else {
                        (
                            [1.0, 5.0, -4.0, 7.0],
                            [0.0, -2.0, 0.0, 6.0],
                            [0.0, 0.0, 3.0, -2.0, 5.0, 0.0, 0.0, 0.0],
                        )
                    };
                    close(
                        &session.tensor_values_lossy_f64(output).unwrap(),
                        &expected,
                        0.0,
                    );
                    let loss = weighted_loss(&mut session, output, &[3.0, -4.0, 5.0, 6.0]);
                    let report = session.tensor_backward(loss).unwrap();
                    if input_grad {
                        close(report.gradient(input).unwrap(), &grad_input, 0.0);
                    } else {
                        assert!(report.gradient(input).is_none());
                    }
                    if src_grad {
                        close(report.gradient(src).unwrap(), &grad_src, 0.0);
                    } else {
                        assert!(report.gradient(src).is_none());
                    }
                }
            }
        }
    }
}

#[test]
fn extrema_keep_exact_selected_values_and_propagate_nonfinite_oracles() {
    let infinity = f64::INFINITY;
    let nan = f64::NAN;
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for reduce in ["amax", "amin"] {
                for create_graph in [false, true] {
                    let mut session = FrankenTorchSession::new(mode);
                    let input = tensor(
                        &mut session,
                        &[infinity, -infinity, 3.0, nan, 0.0, -0.0, 2.0, 1.0],
                        &[8],
                        dtype,
                        true,
                    );
                    let src = tensor(
                        &mut session,
                        &[2.0, -2.0, nan, 5.0, infinity, -infinity, 0.0, 2.0, 2.0],
                        &[9],
                        dtype,
                        true,
                    );
                    let index = tensor(
                        &mut session,
                        &[0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0, 6.0, 6.0],
                        &[9],
                        DType::F64,
                        false,
                    );
                    let output = session
                        .tensor_scatter_reduce(input, 0, index, src, reduce)
                        .unwrap();
                    let (expected, grad_input, grad_src) = if reduce == "amax" {
                        (
                            [infinity, -2.0, nan, nan, infinity, -0.0, 2.0, 1.0],
                            [1.0, 0.0, nan, nan, 0.0, 3.0, 3.0, 7.0],
                            [0.0, 2.0, nan, nan, 5.0, 0.0, 3.0, 3.0, 3.0],
                        )
                    } else {
                        (
                            [2.0, -infinity, nan, nan, -infinity, -0.0, 2.0, 1.0],
                            [0.0, 2.0, nan, nan, 0.0, 3.0, 3.0, 7.0],
                            [1.0, 0.0, nan, nan, 0.0, 5.0, 3.0, 3.0, 3.0],
                        )
                    };
                    let values = session.tensor_values_lossy_f64(output).unwrap();
                    close(&values, &expected, 0.0);
                    assert!(
                        values[5].is_sign_negative(),
                        "select preserves the first zero's sign"
                    );
                    let inference = session
                        .with_no_grad(|s| s.tensor_scatter_reduce(input, 0, index, src, reduce))
                        .unwrap();
                    assert!(!session.tensor_requires_grad(inference).unwrap());
                    close(
                        &session.tensor_values_lossy_f64(inference).unwrap(),
                        &expected,
                        0.0,
                    );
                    let loss = weighted_loss(
                        &mut session,
                        output,
                        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 9.0, 7.0],
                    );
                    let mut options = BackwardOptions::for_mode(mode);
                    options.create_graph = create_graph;
                    let report = session.tensor_backward_with_options(loss, options).unwrap();
                    close(report.gradient(input).unwrap(), &grad_input, 0.0);
                    close(report.gradient(src).unwrap(), &grad_src, 0.0);
                }
            }
        }
    }
}

#[test]
fn extrema_nonfinite_cotangents_follow_mask_after_division() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for create_graph in [false, true] {
                let mut session = FrankenTorchSession::new(mode);
                let input = tensor(&mut session, &[1.0], &[1], dtype, true);
                let src = tensor(&mut session, &[2.0], &[1], dtype, true);
                let index = tensor(&mut session, &[0.0], &[1], DType::F64, false);
                let output = session
                    .tensor_scatter_reduce(input, 0, index, src, "amax")
                    .unwrap();
                let loss = weighted_loss(&mut session, output, &[f64::INFINITY]);
                let mut options = BackwardOptions::for_mode(mode);
                options.create_graph = create_graph;
                let report = session.tensor_backward_with_options(loss, options).unwrap();
                close(report.gradient(input).unwrap(), &[f64::NAN], 0.0);
                close(report.gradient(src).unwrap(), &[f64::INFINITY], 0.0);
            }
        }
    }
}

#[test]
fn extrema_cubic_ties_retain_second_and_third_derivatives() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for reduce in ["amax", "amin"] {
                let sign = if reduce == "amax" { 1.0 } else { -1.0 };
                let mut session = FrankenTorchSession::new(mode);
                let x = tensor(
                    &mut session,
                    &[2.0 * sign, 2.0 * sign, 2.0 * sign, sign],
                    &[4],
                    dtype,
                    true,
                );
                let input = session.tensor_narrow(x, 0, 0, 1).unwrap();
                let src = session.tensor_narrow(x, 0, 1, 3).unwrap();
                let index = tensor(&mut session, &[0.0, 0.0, 0.0], &[3], DType::F64, false);
                let output = session
                    .tensor_scatter_reduce(input, 0, index, src, reduce)
                    .unwrap();
                let square = session.tensor_mul(output, output).unwrap();
                let cube = session.tensor_mul(square, output).unwrap();
                let loss = weighted_loss(&mut session, cube, &[2.0]);
                let mut options = BackwardOptions::for_mode(mode);
                options.create_graph = true;
                options.retain_graph = true;
                let first = session.tensor_backward_with_options(loss, options).unwrap();
                close(first.gradient(x).unwrap(), &[8.0, 8.0, 8.0, 0.0], 1e-10);
                let first_sum = session.tensor_sum(first.gradient_node(x).unwrap()).unwrap();
                let second = session
                    .tensor_backward_with_options(first_sum, options)
                    .unwrap();
                close(
                    second.gradient(x).unwrap(),
                    &[8.0 * sign, 8.0 * sign, 8.0 * sign, 0.0],
                    1e-10,
                );
                let second_sum = session
                    .tensor_sum(second.gradient_node(x).unwrap())
                    .unwrap();
                let third = session.tensor_backward(second_sum).unwrap();
                close(third.gradient(x).unwrap(), &[4.0, 4.0, 4.0, 0.0], 1e-10);
            }
        }
    }
}

#[test]
fn extrema_strided_inputs_and_nonleading_axes_route_gradients() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            let mut session = FrankenTorchSession::new(mode);
            let input_leaf = tensor(&mut session, &[2.0, 7.0, 5.0, 1.0], &[2, 2], dtype, true);
            let source_leaf = tensor(
                &mut session,
                &[2.0, 5.0, 3.0, 7.0, 1.0, 7.0],
                &[3, 2],
                dtype,
                true,
            );
            let index_leaf = tensor(
                &mut session,
                &[0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
                &[3, 2],
                DType::F64,
                false,
            );
            let input = session.tensor_transpose(input_leaf, 0, 1).unwrap();
            let src = session.tensor_transpose(source_leaf, 0, 1).unwrap();
            let index = session.tensor_transpose(index_leaf, 0, 1).unwrap();
            let output = session
                .tensor_scatter_reduce(input, 1, index, src, "amax")
                .unwrap();
            close(
                &session.tensor_values_lossy_f64(output).unwrap(),
                &[3.0, 5.0, 7.0, 5.0],
                0.0,
            );
            let loss = weighted_loss(&mut session, output, &[3.0, 4.0, 6.0, 8.0]);
            let report = session.tensor_backward(loss).unwrap();
            close(
                report.gradient(input_leaf).unwrap(),
                &[0.0, 2.0, 4.0, 0.0],
                0.0,
            );
            close(
                report.gradient(source_leaf).unwrap(),
                &[0.0, 8.0, 3.0, 2.0, 0.0, 2.0],
                0.0,
            );
        }
    }
}

#[test]
fn extrema_monotonic_sources_route_only_the_final_winner() {
    const LENGTH: usize = 8192;
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for reduce in ["amax", "amin"] {
                let mut session = FrankenTorchSession::new(mode);
                let sign = if reduce == "amax" { 1.0 } else { -1.0 };
                let input = tensor(&mut session, &[0.0], &[1], dtype, true);
                let source_values: Vec<f64> =
                    (1..=LENGTH).map(|value| sign * value as f64).collect();
                let src = tensor(&mut session, &source_values, &[LENGTH], dtype, true);
                let index = tensor(
                    &mut session,
                    &vec![0.0; LENGTH],
                    &[LENGTH],
                    DType::F64,
                    false,
                );
                let output = session
                    .tensor_scatter_reduce(input, 0, index, src, reduce)
                    .unwrap();
                close(
                    &session.tensor_values_lossy_f64(output).unwrap(),
                    &[sign * LENGTH as f64],
                    0.0,
                );
                let loss = weighted_loss(&mut session, output, &[-3.0]);
                let report = session.tensor_backward(loss).unwrap();
                close(report.gradient(input).unwrap(), &[0.0], 0.0);
                let mut expected = vec![0.0; LENGTH];
                expected[LENGTH - 1] = -3.0;
                close(report.gradient(src).unwrap(), &expected, 0.0);
            }
        }
    }
}

#[test]
fn scatter_sum_preserves_half_precision_training_and_no_grad() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F16, DType::BF16] {
            let mut session = FrankenTorchSession::new(mode);
            let input_leaf = tensor(&mut session, &[1.0, 2.0, 3.0], &[3], DType::F64, true);
            let src_leaf = tensor(&mut session, &[4.0, 5.0, 6.0], &[3], DType::F64, true);
            let input = session.tensor_to_dtype(input_leaf, dtype).unwrap();
            let src = session.tensor_to_dtype(src_leaf, dtype).unwrap();
            let index = tensor(&mut session, &[0.0, 0.0, 2.0], &[3], DType::F64, false);
            let output = session
                .tensor_scatter_reduce(input, 0, index, src, "sum")
                .unwrap();
            assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
            close(
                &session.tensor_values_lossy_f64(output).unwrap(),
                &[10.0, 2.0, 9.0],
                0.0,
            );
            let untracked = session
                .with_no_grad(|s| s.tensor_scatter_reduce(input, 0, index, src, "sum"))
                .unwrap();
            assert_eq!(session.tensor_dtype(untracked).unwrap(), dtype);
            assert!(!session.tensor_requires_grad(untracked).unwrap());
            close(
                &session.tensor_values_lossy_f64(untracked).unwrap(),
                &[10.0, 2.0, 9.0],
                0.0,
            );
            let output_f64 = session.tensor_to_dtype(output, DType::F64).unwrap();
            let loss = weighted_loss(&mut session, output_f64, &[2.0, 3.0, -4.0]);
            let report = session.tensor_backward(loss).unwrap();
            close(report.gradient(input_leaf).unwrap(), &[2.0, 3.0, -4.0], 0.0);
            close(report.gradient(src_leaf).unwrap(), &[2.0, 2.0, -4.0], 0.0);
        }
    }
}

#[test]
fn mean_f32_training_preserves_dtype_and_weighted_gradients() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            let mut session = FrankenTorchSession::new(mode);
            let input = tensor(&mut session, &[2.0, 5.0, 7.0], &[3], dtype, true);
            let src = tensor(&mut session, &[4.0, 6.0, 8.0], &[3], dtype, true);
            let index = tensor(&mut session, &[0.0, 0.0, 2.0], &[3], DType::F64, false);
            let output = session
                .tensor_scatter_reduce(input, 0, index, src, "mean")
                .unwrap();
            assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
            close(
                &session.tensor_values_lossy_f64(output).unwrap(),
                &[4.0, 5.0, 7.5],
                0.0,
            );
            let loss = weighted_loss(&mut session, output, &[3.0, 4.0, 10.0]);
            let mut options = BackwardOptions::for_mode(mode);
            options.create_graph = true;
            let report = session.tensor_backward_with_options(loss, options).unwrap();
            close(report.gradient(input).unwrap(), &[1.0, 4.0, 5.0], 1e-12);
            close(report.gradient(src).unwrap(), &[1.0, 1.0, 5.0], 1e-12);
        }
    }
}

#[test]
fn scatter_reduce_empty_axes_and_empty_sources_preserve_gradients() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for reduce in ["sum", "prod", "mean", "amax", "amin"] {
                for grad in [false, true] {
                    for shape in [&[2, 0][..], &[0, 3][..]] {
                        let mut session = FrankenTorchSession::new(mode);
                        let input = tensor(&mut session, &[], shape, dtype, grad);
                        let src = tensor(&mut session, &[], shape, dtype, grad);
                        let index = tensor(&mut session, &[], shape, DType::F64, false);
                        let output = session
                            .tensor_scatter_reduce(input, 1, index, src, reduce)
                            .unwrap();
                        assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
                        assert_eq!(session.tensor_shape(output).unwrap(), shape);
                        assert!(session.tensor_values_lossy_f64(output).unwrap().is_empty());
                        if grad {
                            let loss = session.tensor_sum(output).unwrap();
                            let report = session.tensor_backward(loss).unwrap();
                            assert!(report.gradient(input).unwrap().is_empty());
                            assert!(report.gradient(src).unwrap().is_empty());
                        }
                    }
                }
                let mut session = FrankenTorchSession::new(mode);
                let input = tensor(&mut session, &[1.0, 2.0], &[2], dtype, true);
                let src = tensor(&mut session, &[], &[0], dtype, true);
                let index = tensor(&mut session, &[], &[0], DType::F64, false);
                let output = session
                    .tensor_scatter_reduce(input, 0, index, src, reduce)
                    .unwrap();
                let loss = weighted_loss(&mut session, output, &[2.0, -3.0]);
                let report = session.tensor_backward(loss).unwrap();
                close(report.gradient(input).unwrap(), &[2.0, -3.0], 0.0);
                assert!(report.gradient(src).unwrap().is_empty());
            }
        }
    }
}

#[test]
fn scatter_reduce_rejects_invalid_metadata_before_recording_nodes() {
    for reduce in ["sum", "prod", "mean", "amax", "amin"] {
        for invalid_index in [f64::NAN, f64::INFINITY, 0.5, 3.0, -4.0] {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let input = tensor(&mut session, &[1.0, 2.0, 3.0], &[3], DType::F32, true);
            let src = tensor(&mut session, &[4.0], &[1], DType::F32, true);
            let index = tensor(&mut session, &[invalid_index], &[1], DType::F64, false);
            let count = session.autograd_graph_node_count();
            assert!(
                session
                    .tensor_scatter_reduce(input, 0, index, src, reduce)
                    .is_err()
            );
            assert_eq!(session.autograd_graph_node_count(), count);
        }
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let input = tensor(&mut session, &[1.0], &[1], DType::F32, true);
        let src = tensor(&mut session, &[2.0], &[1], DType::F64, true);
        let index = tensor(&mut session, &[0.0], &[1], DType::F64, false);
        let count = session.autograd_graph_node_count();
        assert!(
            session
                .tensor_scatter_reduce(input, 0, index, src, reduce)
                .is_err()
        );
        assert_eq!(session.autograd_graph_node_count(), count);
    }
}

#[test]
fn extrema_backward_rejects_mutated_saved_input_source_and_index() {
    for mutate in 0..3 {
        for create_graph in [false, true] {
            let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
            let input = tensor(&mut session, &[1.0, 4.0], &[2], DType::F64, true);
            let src = tensor(&mut session, &[2.0], &[1], DType::F64, true);
            let index = tensor(&mut session, &[0.0], &[1], DType::F64, false);
            let output = session
                .tensor_scatter_reduce(input, 0, index, src, "amax")
                .unwrap();
            let loss = session.tensor_sum(output).unwrap();
            session
                .with_no_grad(|s| s.tensor_fill_([input, src, index][mutate], 1.0))
                .unwrap();
            let mut options = BackwardOptions::for_mode(ExecutionMode::Strict);
            options.create_graph = create_graph;
            let error = session
                .tensor_backward_with_options(loss, options)
                .unwrap_err();
            assert!(format!("{error:?}").contains("Version"), "{error:?}");
        }
    }
}
