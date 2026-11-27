//! Exercise autocast, checkpoint replay, and typed training-state restoration.
//! Analytic AMP and VJP fixtures use values exactly representable in F16 and
//! BF16, while state fixtures check raw storage bits, identity, and atomicity.

use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::{DType, ExecutionMode};

fn weighted_loss(
    session: &mut FrankenTorchSession,
    output: TensorNodeId,
    weights: Vec<f32>,
) -> TensorNodeId {
    let shape = session.tensor_shape(output).unwrap();
    let output_f32 = session.tensor_to_dtype(output, DType::F32).unwrap();
    let weights = session.tensor_variable_f32(weights, shape, false).unwrap();
    let weighted = session.tensor_mul(output_f32, weights).unwrap();
    session.tensor_sum(weighted).unwrap()
}

#[test]
fn autocast_matmul_keeps_f32_parameters_and_backpropagates_weighted_loss() {
    for dtype in [DType::F16, DType::BF16] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(vec![1., 2., 3., 4.], vec![2, 2], true)
            .unwrap();
        let w = s
            .tensor_variable_f32(vec![2., -1., 0., 3.], vec![2, 2], true)
            .unwrap();
        let y = s.with_autocast(dtype, |s| s.tensor_matmul(x, w).unwrap());
        assert_eq!(s.tensor_dtype(y).unwrap(), dtype);
        assert_eq!(s.tensor_dtype(x).unwrap(), DType::F32);
        assert_eq!(s.tensor_dtype(w).unwrap(), DType::F32);
        assert_eq!(s.tensor_values_lossy_f64(y).unwrap(), [2., 5., 6., 9.]);
        let loss = weighted_loss(&mut s, y, vec![1., 2., 3., 4.]);
        let report = s.tensor_backward(loss).unwrap();
        assert_eq!(s.tensor_gradient(&report, x).unwrap(), [0., 6., 2., 12.]);
        assert_eq!(s.tensor_gradient(&report, w).unwrap(), [10., 14., 14., 20.]);
    }
}

#[test]
fn autocast_linear_casts_bias_and_preserves_all_parameter_gradients() {
    for dtype in [DType::F16, DType::BF16] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(vec![1., 2., 3., 4.], vec![2, 2], true)
            .unwrap();
        let w = s
            .tensor_variable_f32(vec![2., -1., 0., 3.], vec![2, 2], true)
            .unwrap();
        let b = s
            .tensor_variable_f32(vec![0.5, -0.5], vec![2], true)
            .unwrap();
        let y = s.with_autocast(dtype, |s| s.functional_linear(x, w, Some(b)).unwrap());
        assert_eq!(s.tensor_dtype(y).unwrap(), dtype);
        assert_eq!(s.tensor_dtype(b).unwrap(), DType::F32);
        assert_eq!(s.tensor_values_lossy_f64(y).unwrap(), [0.5, 5.5, 2.5, 11.5]);
        let loss = weighted_loss(&mut s, y, vec![1., 2., 3., 4.]);
        let report = s.tensor_backward(loss).unwrap();
        assert_eq!(s.tensor_gradient(&report, x).unwrap(), [2., 5., 6., 9.]);
        assert_eq!(s.tensor_gradient(&report, w).unwrap(), [10., 14., 14., 20.]);
        assert_eq!(s.tensor_gradient(&report, b).unwrap(), [4., 6.]);
    }
}

#[test]
fn autocast_conv2d_backpropagates_nonuniform_spatial_loss() {
    for dtype in [DType::F16, DType::BF16] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32((1..=9).map(|v| v as f32).collect(), vec![1, 1, 3, 3], true)
            .unwrap();
        let w = s
            .tensor_variable_f32(vec![1., 0., 0., -1.], vec![1, 1, 2, 2], true)
            .unwrap();
        let b = s.tensor_variable_f32(vec![0.5], vec![1], true).unwrap();
        let y = s.with_autocast(dtype, |s| {
            s.functional_conv2d(x, w, Some(b), (1, 1), (0, 0)).unwrap()
        });
        assert_eq!(s.tensor_dtype(y).unwrap(), dtype);
        assert_eq!(s.tensor_values_lossy_f64(y).unwrap(), [-3.5; 4]);
        let loss = weighted_loss(&mut s, y, vec![1., 2., 3., 4.]);
        let report = s.tensor_backward(loss).unwrap();
        assert_eq!(
            s.tensor_gradient(&report, x).unwrap(),
            [1., 2., 0., 3., 3., -2., 0., -3., -4.]
        );
        assert_eq!(s.tensor_gradient(&report, w).unwrap(), [37., 47., 67., 77.]);
        assert_eq!(s.tensor_gradient(&report, b).unwrap(), [10.]);
    }
}

#[test]
fn autocast_conv1d_and_conv3d_reach_low_precision_execution() {
    for dtype in [DType::F16, DType::BF16] {
        for spatial_dims in [1, 3] {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let mut shape = vec![1; spatial_dims + 2];
            shape[spatial_dims + 1] = 4;
            let x = s
                .tensor_variable_f32(vec![1., 2., 3., 4.], shape, true)
                .unwrap();
            let mut kernel_shape = vec![1; spatial_dims + 2];
            kernel_shape[spatial_dims + 1] = 2;
            let w = s
                .tensor_variable_f32(vec![2., -1.], kernel_shape, true)
                .unwrap();
            let b = s.tensor_variable_f32(vec![0.5], vec![1], true).unwrap();
            let y = s.with_autocast(dtype, |s| {
                if spatial_dims == 1 {
                    s.functional_conv1d(x, w, Some(b), 1, 0).unwrap()
                } else {
                    s.functional_conv3d(x, w, Some(b), (1, 1, 1), (0, 0, 0))
                        .unwrap()
                }
            });
            assert_eq!(s.tensor_dtype(y).unwrap(), dtype);
            assert_eq!(s.tensor_values_lossy_f64(y).unwrap(), [0.5, 1.5, 2.5]);
            let loss = weighted_loss(&mut s, y, vec![1., 2., 3.]);
            let report = s.tensor_backward(loss).unwrap();
            assert_eq!(s.tensor_gradient(&report, x).unwrap(), [2., 3., 4., -3.]);
            assert_eq!(s.tensor_gradient(&report, w).unwrap(), [14., 20.]);
            assert_eq!(s.tensor_gradient(&report, b).unwrap(), [6.]);
        }
    }
}

#[test]
fn autocast_grouped_dilated_convolution_preserves_channel_gradient_routing() {
    for dtype in [DType::F16, DType::BF16] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32((1..=10).map(|v| v as f32).collect(), vec![1, 2, 5], true)
            .unwrap();
        let w = s
            .tensor_variable_f32(vec![1., -1., 2., 1.], vec![2, 1, 2], true)
            .unwrap();
        let b = s.tensor_variable_f32(vec![0., 0.5], vec![2], true).unwrap();
        let y = s.with_autocast(dtype, |s| {
            s.functional_conv1d_dilated(x, w, Some(b), 1, 0, 2, 2)
                .unwrap()
        });
        assert_eq!(s.tensor_dtype(y).unwrap(), dtype);
        assert_eq!(
            s.tensor_values_lossy_f64(y).unwrap(),
            [-2., -2., -2., 20.5, 23.5, 26.5]
        );
        let loss = weighted_loss(&mut s, y, vec![1.; 6]);
        let report = s.tensor_backward(loss).unwrap();
        assert_eq!(
            s.tensor_gradient(&report, x).unwrap(),
            [1., 1., 0., -1., -1., 2., 2., 3., 1., 1.]
        );
        assert_eq!(s.tensor_gradient(&report, w).unwrap(), [6., 12., 21., 27.]);
        assert_eq!(s.tensor_gradient(&report, b).unwrap(), [3., 3.]);
    }
}

#[test]
fn autocast_batched_and_affine_matmuls_preserve_half_outputs() {
    for dtype in [DType::F16, DType::BF16] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(vec![1., 2., 3., 4.], vec![1, 2, 2], false)
            .unwrap();
        let w = s
            .tensor_variable_f32(vec![2., -1., 0., 3.], vec![1, 2, 2], false)
            .unwrap();
        let b = s
            .tensor_variable_f32(vec![1.; 4], vec![2, 2], false)
            .unwrap();
        s.with_autocast(dtype, |s| {
            let y = s.tensor_bmm(x, w).unwrap();
            assert_eq!(s.tensor_dtype(y).unwrap(), dtype);
            assert_eq!(s.tensor_values_lossy_f64(y).unwrap(), [2., 5., 6., 9.]);
            let y = s.tensor_baddbmm(b, x, w, 0.5, 2.).unwrap();
            assert_eq!(s.tensor_dtype(y).unwrap(), dtype);
            assert_eq!(
                s.tensor_values_lossy_f64(y).unwrap(),
                [4.5, 10.5, 12.5, 18.5]
            );
            let y = s.tensor_addbmm(b, x, w, 0.5, 2.).unwrap();
            assert_eq!(s.tensor_dtype(y).unwrap(), dtype);
            assert_eq!(
                s.tensor_values_lossy_f64(y).unwrap(),
                [4.5, 10.5, 12.5, 18.5]
            );
        });
    }
}

#[test]
fn autocast_nested_precision_and_disabled_regions_restore_parent_policy() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable_f32(vec![2.], vec![1, 1], true).unwrap();
    s.with_autocast(DType::F16, |s| {
        let h = s.tensor_matmul(x, x).unwrap();
        assert_eq!(s.tensor_dtype(h).unwrap(), DType::F16);
        let b = s.with_autocast(DType::BF16, |s| s.tensor_matmul(h, x).unwrap());
        assert_eq!(s.tensor_dtype(b).unwrap(), DType::BF16);
        assert_eq!(s.autocast_dtype(), Some(DType::F16));
        let f = s.with_autocast_disabled(|s| {
            assert!(!s.is_autocast_enabled());
            let b = s.tensor_to_dtype(b, DType::F32).unwrap();
            s.tensor_matmul(b, x).unwrap()
        });
        assert_eq!(s.tensor_dtype(f).unwrap(), DType::F32);
        assert_eq!(s.tensor_values_f32(f).unwrap(), [16.]);
        assert_eq!(s.autocast_dtype(), Some(DType::F16));
        let loss = s.tensor_sum(f).unwrap();
        let report = s.tensor_backward(loss).unwrap();
        assert_eq!(s.tensor_gradient(&report, x).unwrap(), [32.]);
    });
    assert!(!s.is_autocast_enabled());
}

#[test]
fn autocast_preserves_f64_and_explicit_noneligible_operations() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let value = 1. + 2_f64.powi(-40);
    let x = s.tensor_variable(vec![value], vec![1, 1], true).unwrap();
    let one = s.tensor_variable(vec![1.], vec![1, 1], false).unwrap();
    let f = s.tensor_variable_f32(vec![1.25], vec![1], true).unwrap();
    s.with_autocast(DType::BF16, |s| {
        let y = s.tensor_matmul(x, one).unwrap();
        assert_eq!(s.tensor_dtype(y).unwrap(), DType::F64);
        assert_eq!(s.tensor_values(y).unwrap(), [value]);
        let added = s.tensor_add(f, f).unwrap();
        assert_eq!(s.tensor_dtype(added).unwrap(), DType::F32);
    });
}

#[test]
fn all_floating_cast_pairs_are_differentiable() {
    for source in [DType::F64, DType::F32, DType::F16, DType::BF16] {
        for target in [DType::F64, DType::F32, DType::F16, DType::BF16] {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable(vec![1., -2.5, 4.], vec![3], true)
                .unwrap();
            let src = s.tensor_to_dtype(x, source).unwrap();
            let y = s.tensor_to_dtype(src, target).unwrap();
            assert_eq!(s.tensor_dtype(y).unwrap(), target);
            assert_eq!(s.tensor_values_lossy_f64(y).unwrap(), [1., -2.5, 4.]);
            let loss = weighted_loss(&mut s, y, vec![1., 2., 3.]);
            let report = s.tensor_backward(loss).unwrap();
            assert_eq!(s.tensor_gradient(&report, x).unwrap(), [1., 2., 3.]);
        }
    }
}

#[test]
fn autocast_does_not_change_in_place_matrix_precision() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s
        .tensor_variable_f32(vec![1.001], vec![1, 1], false)
        .unwrap();
    let one = s.tensor_variable_f32(vec![1.], vec![1, 1], false).unwrap();
    let out = s.tensor_variable_f32(vec![0.], vec![1, 1], false).unwrap();
    let xb = s.tensor_reshape(x, vec![1, 1, 1]).unwrap();
    let oneb = s.tensor_reshape(one, vec![1, 1, 1]).unwrap();
    let outb = s
        .tensor_variable_f32(vec![0.], vec![1, 1, 1], false)
        .unwrap();
    s.with_autocast(DType::BF16, |s| {
        s.tensor_addmm_(out, x, one, 0., 1.).unwrap();
        assert_eq!(s.tensor_dtype(out).unwrap(), DType::F32);
        assert_eq!(s.tensor_values_f32(out).unwrap(), [1.001]);
        s.tensor_baddbmm_(outb, xb, oneb, 0., 1.).unwrap();
        assert_eq!(s.tensor_dtype(outb).unwrap(), DType::F32);
        assert_eq!(s.tensor_values_f32(outb).unwrap(), [1.001]);
        s.tensor_addbmm_(out, xb, oneb, 0., 1.).unwrap();
        assert_eq!(s.tensor_dtype(out).unwrap(), DType::F32);
        assert_eq!(s.tensor_values_f32(out).unwrap(), [1.001]);
        assert_eq!(s.autocast_dtype(), Some(DType::BF16));
    });
}

#[test]
fn autocast_checkpoint_convolution_matches_training_and_inference() {
    for dtype in [DType::F16, DType::BF16] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32((1..=9).map(|v| v as f32).collect(), vec![1, 1, 3, 3], true)
            .unwrap();
        let w = s
            .tensor_variable_f32(vec![1., 0., 0., -1.], vec![1, 1, 2, 2], true)
            .unwrap();
        let b = s.tensor_variable_f32(vec![0.5], vec![1], true).unwrap();
        let y = s.with_autocast(dtype, |s| {
            s.checkpoint(&[x, w, b], |s, inputs| {
                s.functional_conv2d(inputs[0], inputs[1], Some(inputs[2]), (1, 1), (0, 0))
            })
            .unwrap()
        });
        assert_eq!(s.tensor_dtype(y).unwrap(), dtype);
        assert_eq!(s.tensor_values_lossy_f64(y).unwrap(), [-3.5; 4]);
        let loss = weighted_loss(&mut s, y, vec![1., 2., 3., 4.]);
        let report = s.tensor_backward(loss).unwrap();
        assert_eq!(s.tensor_gradient(&report, w).unwrap(), [37., 47., 67., 77.]);
        assert_eq!(s.tensor_gradient(&report, b).unwrap(), [10.]);
        let inference = s.with_no_grad(|s| {
            s.with_autocast(dtype, |s| {
                s.functional_conv2d(x, w, Some(b), (1, 1), (0, 0)).unwrap()
            })
        });
        assert_eq!(s.tensor_dtype(inference).unwrap(), dtype);
        assert!(!s.tensor_requires_grad(inference).unwrap());
        assert_eq!(s.tensor_values_lossy_f64(inference).unwrap(), [-3.5; 4]);
    }
}

#[test]
fn autocast_attention_has_same_dtype_in_inference_training_and_checkpoint() {
    for dtype in [DType::F16, DType::BF16] {
        for tensor_api in [false, true] {
            for checkpointed in [false, true] {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let q = s
                    .tensor_variable_f32(vec![0.; 2], vec![1, 2], true)
                    .unwrap();
                let k = s
                    .tensor_variable_f32(vec![0.; 4], vec![2, 2], true)
                    .unwrap();
                let v = s
                    .tensor_variable_f32(vec![2., 4.], vec![2, 1], true)
                    .unwrap();
                let attention = move |s: &mut FrankenTorchSession, inputs: &[TensorNodeId]| {
                    if tensor_api {
                        s.tensor_scaled_dot_product_attention(
                            inputs[0], inputs[1], inputs[2], None, false, None,
                        )
                    } else {
                        s.scaled_dot_product_attention(
                            inputs[0], inputs[1], inputs[2], None, 0., false,
                        )
                    }
                };
                let y = s.with_autocast(dtype, |s| {
                    if checkpointed {
                        s.checkpoint(&[q, k, v], attention).unwrap()
                    } else {
                        attention(s, &[q, k, v]).unwrap()
                    }
                });
                assert_eq!(s.tensor_dtype(y).unwrap(), dtype);
                assert_eq!(s.tensor_values_lossy_f64(y).unwrap(), [3.]);
                let loss = weighted_loss(&mut s, y, vec![2.]);
                let report = s.tensor_backward(loss).unwrap();
                assert_eq!(s.tensor_gradient(&report, q).unwrap(), [0.; 2]);
                assert_eq!(s.tensor_gradient(&report, k).unwrap(), [0.; 4]);
                assert_eq!(s.tensor_gradient(&report, v).unwrap(), [1., 1.]);
                let inference = s.with_no_grad(|s| {
                    s.with_autocast(dtype, |s| attention(s, &[q, k, v]).unwrap())
                });
                assert_eq!(s.tensor_dtype(inference).unwrap(), dtype);
                assert!(!s.tensor_requires_grad(inference).unwrap());
                assert_eq!(s.tensor_values_lossy_f64(inference).unwrap(), [3.]);
            }
        }
    }
}

#[test]
fn autocast_rounds_inputs_and_restores_policy_after_disabled_region_unwinds() {
    for (dtype, value) in [(DType::F16, 1.0003_f32), (DType::BF16, 1.003_f32)] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(vec![value], vec![1, 1], false)
            .unwrap();
        let one = s.tensor_variable_f32(vec![1.], vec![1, 1], false).unwrap();
        s.with_autocast(dtype, |s| {
            let y = s.tensor_matmul(x, one).unwrap();
            assert_eq!(s.tensor_values_lossy_f64(y).unwrap(), [1.]);
            assert_eq!(s.tensor_values_f32(x).unwrap(), [value]);
            let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                s.with_autocast_disabled(|s| {
                    assert!(!s.is_autocast_enabled());
                    std::panic::resume_unwind(Box::new("autocast restoration probe"));
                });
            }));
            assert!(unwind.is_err());
            assert_eq!(s.autocast_dtype(), Some(dtype));
        });
        assert!(!s.is_autocast_enabled());
    }
}

#[test]
fn autocast_batched_linear_and_dropout_keep_half_activations() {
    for dtype in [DType::F16, DType::BF16] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(vec![1.; 4], vec![1, 2, 2], true)
            .unwrap();
        let w = s
            .tensor_variable_f32(vec![1., 0., 0., 1.], vec![2, 2], false)
            .unwrap();
        let b = s.tensor_variable_f32(vec![0.; 2], vec![2], true).unwrap();
        let y = s.with_autocast(dtype, |s| {
            let linear = s.functional_linear(x, w, Some(b)).unwrap();
            assert_eq!(s.tensor_shape(linear).unwrap(), [1, 2, 2]);
            assert_eq!(s.tensor_dtype(linear).unwrap(), dtype);
            s.functional_dropout(linear, 0.5, true).unwrap()
        });
        assert_eq!(s.tensor_dtype(y).unwrap(), dtype);
        let values = s.tensor_values_lossy_f64(y).unwrap();
        assert!(values.iter().all(|&v| v == 0. || v == 2.));
        let expected: Vec<_> = values
            .iter()
            .enumerate()
            .map(|(i, v)| v * (i + 1) as f64)
            .collect();
        let loss = weighted_loss(&mut s, y, vec![1., 2., 3., 4.]);
        let report = s.tensor_backward(loss).unwrap();
        assert_eq!(s.tensor_gradient(&report, x).unwrap(), expected);
        assert_eq!(
            s.tensor_gradient(&report, b).unwrap(),
            [expected[0] + expected[2], expected[1] + expected[3]]
        );
    }
}

#[test]
fn autocast_mse_avoids_half_overflow_and_supports_scaled_backward() {
    for dtype in [DType::F16, DType::BF16] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s.tensor_variable_f32(vec![300.], vec![1, 1], true).unwrap();
        let w = s.tensor_variable_f32(vec![1.], vec![1, 1], false).unwrap();
        let target = s.tensor_variable_f32(vec![0.], vec![1, 1], true).unwrap();
        let target_half = s.tensor_to_dtype(target, dtype).unwrap();
        let loss = s.with_autocast(dtype, |s| {
            let prediction = s.tensor_matmul(x, w).unwrap();
            assert_eq!(s.tensor_dtype(prediction).unwrap(), dtype);
            s.tensor_mse_loss(prediction, target_half, "mean").unwrap()
        });
        assert_eq!(s.tensor_dtype(loss).unwrap(), DType::F32);
        assert_eq!(s.tensor_values_f32(loss).unwrap(), [90_000.]);
        let scaled = s.tensor_mul_scalar(loss, 128.).unwrap();
        let report = s.tensor_backward(scaled).unwrap();
        assert_eq!(s.tensor_gradient(&report, x).unwrap(), [76_800.]);
        assert_eq!(s.tensor_gradient(&report, target).unwrap(), [-76_800.]);
        assert_eq!(s.tensor_dtype(x).unwrap(), DType::F32);
    }
}

#[test]
fn autocast_regression_losses_use_f32_values_and_gradients() {
    for dtype in [DType::F16, DType::BF16] {
        for kind in [0, 1, 2] {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(vec![-2., 0.5, 2.], vec![3], true)
                .unwrap();
            let t = s.tensor_variable_f32(vec![0.; 3], vec![3], false).unwrap();
            let xh = s.tensor_to_dtype(x, dtype).unwrap();
            let th = s.tensor_to_dtype(t, dtype).unwrap();
            let loss = s.with_autocast(dtype, |s| match kind {
                0 => s.tensor_l1_loss(xh, th, "sum").unwrap(),
                1 => s.tensor_smooth_l1_loss(xh, th, "sum", 1.).unwrap(),
                _ => s.tensor_huber_loss(xh, th, "sum", 2.).unwrap(),
            });
            assert_eq!(s.tensor_dtype(loss).unwrap(), DType::F32);
            assert_eq!(
                s.tensor_values_f32(loss).unwrap(),
                [[4.5], [3.125], [4.125]][kind]
            );
            let report = s.tensor_backward(loss).unwrap();
            assert_eq!(
                s.tensor_gradient(&report, x).unwrap(),
                [[-1., 1., 1.], [-1., 0.5, 1.], [-2., 0.5, 2.]][kind]
            );
        }
    }
}

#[test]
fn autocast_binary_classification_losses_use_f32() {
    for dtype in [DType::F16, DType::BF16] {
        for weighted in [false, true] {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(vec![-100., 100.], vec![2], true)
                .unwrap();
            let t = s.tensor_variable_f32(vec![1., 0.], vec![2], false).unwrap();
            let xh = s.tensor_to_dtype(x, dtype).unwrap();
            let th = s.tensor_to_dtype(t, dtype).unwrap();
            let loss = s.with_autocast(dtype, |s| {
                if weighted {
                    s.tensor_bce_with_logits_pos_weight(xh, th, 2., "sum")
                        .unwrap()
                } else {
                    s.tensor_bce_with_logits_loss(xh, th, "sum").unwrap()
                }
            });
            assert_eq!(s.tensor_dtype(loss).unwrap(), DType::F32);
            assert_eq!(
                s.tensor_values_f32(loss).unwrap(),
                [if weighted { 300. } else { 200. }]
            );
            let report = s.tensor_backward(loss).unwrap();
            assert_eq!(
                s.tensor_gradient(&report, x).unwrap(),
                [if weighted { -2. } else { -1. }, 1.]
            );
        }
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(vec![0.25, 0.75], vec![2], true)
            .unwrap();
        let t = s.tensor_variable_f32(vec![0., 1.], vec![2], false).unwrap();
        let xh = s.tensor_to_dtype(x, dtype).unwrap();
        let th = s.tensor_to_dtype(t, dtype).unwrap();
        let loss = s.with_autocast(dtype, |s| s.tensor_bce_loss(xh, th, "sum").unwrap());
        assert_eq!(s.tensor_dtype(loss).unwrap(), DType::F32);
        assert!((s.tensor_values_f32(loss).unwrap()[0] + 2. * 0.75_f32.ln()).abs() < 1e-6);
        let report = s.tensor_backward(loss).unwrap();
        for (&actual, expected) in s
            .tensor_gradient(&report, x)
            .unwrap()
            .iter()
            .zip([4. / 3., -4. / 3.])
        {
            assert!((actual - expected).abs() < 1e-6);
        }
    }
}

#[test]
fn autocast_cross_entropy_handles_indices_soft_labels_weights_and_smoothing() {
    for dtype in [DType::F16, DType::BF16] {
        for kind in [0, 1, 2, 3] {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let x = s
                .tensor_variable_f32(vec![1000., -1000., -1000., 1000.], vec![2, 2], true)
                .unwrap();
            let xh = s.tensor_to_dtype(x, dtype).unwrap();
            let target = s.tensor_variable(vec![1., 0.], vec![2], false).unwrap();
            let soft = s
                .tensor_variable_f32(vec![0., 1., 1., 0.], vec![2, 2], false)
                .unwrap();
            let soft = s.tensor_to_dtype(soft, dtype).unwrap();
            let weights = s.tensor_variable_f32(vec![2., 3.], vec![2], false).unwrap();
            let weights = s.tensor_to_dtype(weights, dtype).unwrap();
            let loss = s.with_autocast(dtype, |s| match kind {
                0 => s.tensor_cross_entropy(xh, target, "sum").unwrap(),
                1 => s.tensor_cross_entropy(xh, soft, "sum").unwrap(),
                2 => s
                    .tensor_cross_entropy_full(xh, target, Some(weights), None, "mean")
                    .unwrap(),
                _ => s
                    .tensor_cross_entropy_label_smoothing(xh, target, 0.25, "sum")
                    .unwrap(),
            });
            assert_eq!(s.tensor_dtype(loss).unwrap(), DType::F32);
            assert_eq!(
                s.tensor_values_f32(loss).unwrap(),
                [[4000.], [4000.], [2000.], [3500.]][kind]
            );
            assert_eq!(s.tensor_dtype(target).unwrap(), DType::F64);
            let report = s.tensor_backward(loss).unwrap();
            let expected = [
                [1., -1., -1., 1.],
                [1., -1., -1., 1.],
                [0.6, -0.6, -0.4, 0.4],
                [0.875, -0.875, -0.875, 0.875],
            ][kind];
            for (&actual, expected) in s.tensor_gradient(&report, x).unwrap().iter().zip(expected) {
                assert!(
                    (actual - expected).abs() < 1e-6,
                    "loss {kind}: {actual} != {expected}"
                );
            }
        }
    }
}

#[test]
fn autocast_losses_preserve_explicit_f64_inputs() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let value = 1. + 2_f64.powi(-40);
    let x = s.tensor_variable(vec![value], vec![1], true).unwrap();
    let t = s.tensor_variable(vec![0.], vec![1], false).unwrap();
    let loss = s.with_autocast(DType::F16, |s| s.tensor_mse_loss(x, t, "sum").unwrap());
    assert_eq!(s.tensor_dtype(loss).unwrap(), DType::F64);
    assert_eq!(s.tensor_values(loss).unwrap(), [value * value]);
    let report = s.tensor_backward(loss).unwrap();
    assert_eq!(s.tensor_gradient(&report, x).unwrap(), [2. * value]);
}

#[test]
fn checkpoint_weighted_rectangular_matmul_has_exact_vjp() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s
        .tensor_variable(vec![1., 2., 3., 4., 5., 6.], vec![2, 3], true)
        .unwrap();
    let w = s
        .tensor_variable(vec![1., 2., 3., 4., 5., 6.], vec![3, 2], true)
        .unwrap();
    let out = s
        .checkpoint(&[x, w], |s, inputs| s.tensor_matmul(inputs[0], inputs[1]))
        .unwrap();
    let gradients = s
        .tensor_autograd_grad(
            &[out],
            &[x, w],
            Some(&[vec![1., -2., 3., 4.]]),
            false,
            false,
        )
        .unwrap();
    assert_eq!(
        gradients[0].as_deref(),
        Some(&[-3., -5., -7., 11., 25., 39.][..])
    );
    assert_eq!(
        gradients[1].as_deref(),
        Some(&[13., 14., 17., 16., 21., 18.][..])
    );
}

#[test]
fn checkpoint_reduction_scales_every_input_gradient() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(vec![1., 2., 3.], vec![3], true).unwrap();
    let out = s
        .checkpoint(&[x], |s, inputs| {
            let square = s.tensor_mul(inputs[0], inputs[0])?;
            s.tensor_sum(square)
        })
        .unwrap();
    let loss = s.tensor_mul_scalar(out, 3.).unwrap();
    let report = s.tensor_backward(loss).unwrap();
    assert_eq!(report.gradient(x), Some(&[6., 12., 18.][..]));
}

#[test]
fn checkpoint_broadcast_keeps_frozen_inputs_frozen() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(vec![2., 3.], vec![2, 1], true).unwrap();
    let w = s
        .tensor_variable(vec![5., 7., 11.], vec![1, 3], true)
        .unwrap();
    let bias = s.tensor_variable(vec![1.; 6], vec![2, 3], false).unwrap();
    let out = s
        .checkpoint(&[x, w, bias], |s, inputs| {
            assert!(!s.tensor_requires_grad(inputs[2])?);
            let product = s.tensor_mul(inputs[0], inputs[1])?;
            s.tensor_add(product, inputs[2])
        })
        .unwrap();
    let report = s
        .tensor_autograd_grad_report(
            &[out],
            &[x, w],
            Some(&[vec![1., 2., 3., 4., 5., 6.]]),
            false,
            false,
        )
        .unwrap();
    assert_eq!(report.gradient(x), Some(&[52., 121.][..]));
    assert_eq!(report.gradient(w), Some(&[14., 19., 24.][..]));
    assert!(report.gradient(bias).is_none());
}

#[test]
fn checkpoint_rejects_replay_after_trainable_or_frozen_input_changes() {
    use ft_autograd::AutogradError;
    use ft_core::{DenseTensor, Device};

    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for change_frozen in [false, true] {
            let mut s = FrankenTorchSession::new(mode);
            let x = s.tensor_variable(vec![2., 3.], vec![2], true).unwrap();
            let w = s.tensor_variable(vec![5., 7.], vec![2], false).unwrap();
            let loss = s
                .checkpoint(&[x, w], |s, inputs| {
                    let product = s.tensor_mul(inputs[0], inputs[1])?;
                    s.tensor_sum(product)
                })
                .unwrap();
            let changed = if change_frozen { w } else { x };
            let recorded = s.tensor_snapshot(changed).unwrap().version();
            let new_value =
                DenseTensor::from_contiguous_values(vec![11., 13.], vec![2], Device::Cpu).unwrap();
            s.tensor_load_state(&[(changed, new_value)]).unwrap();
            assert!(matches!(
                s.tensor_backward(loss),
                Err(AutogradError::BorrowedInputVersionMismatch {
                    node, recorded: saved, observed,
                }) if node == changed && saved == recorded && observed == recorded + 1
            ));
        }
    }
}

#[test]
fn checkpoint_complex_vjp_preserves_real_and_imaginary_gradients() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let real = s.tensor_variable(vec![1., 2.], vec![2], true).unwrap();
    let imag = s.tensor_variable(vec![3., 4.], vec![2], true).unwrap();
    let z = s.tensor_complex(real, imag).unwrap();
    let out = s
        .checkpoint(&[z], |s, inputs| s.tensor_conj(inputs[0]))
        .unwrap();
    assert_eq!(s.tensor_dtype(out).unwrap(), DType::Complex128);
    let components = s.tensor_view_as_real(out).unwrap();
    assert_eq!(s.tensor_values(components).unwrap(), [1., -3., 2., -4.]);
    let weights = s
        .tensor_variable(vec![2., -3., 5., 7.], vec![2, 2], false)
        .unwrap();
    let weighted = s.tensor_mul(components, weights).unwrap();
    let loss = s.tensor_sum(weighted).unwrap();
    let report = s.tensor_backward(loss).unwrap();
    assert_eq!(report.gradient(real), Some(&[2., 5.][..]));
    assert_eq!(report.gradient(imag), Some(&[3., -7.][..]));
}

#[test]
fn checkpoint_rng_replay_leaves_parent_stream_unchanged() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    s.rand(vec![7], false).unwrap();
    let x = s.tensor_variable(vec![1.; 16], vec![16], true).unwrap();
    let mut reference = s.clone();
    let direct = reference.functional_dropout(x, 0.5, true).unwrap();
    let replayed = s
        .checkpoint(&[x], |s, inputs| s.functional_dropout(inputs[0], 0.5, true))
        .unwrap();
    assert_eq!(
        s.tensor_values(replayed).unwrap(),
        reference.tensor_values(direct).unwrap()
    );
    let weights = vec![(1..=16).map(f64::from).collect()];
    let expected = reference
        .tensor_autograd_grad(&[direct], &[x], Some(&weights), false, false)
        .unwrap();
    let actual = s
        .tensor_autograd_grad(&[replayed], &[x], Some(&weights), false, false)
        .unwrap();
    assert_eq!(actual, expected);
    let next = s.rand(vec![8], false).unwrap();
    let reference_next = reference.rand(vec![8], false).unwrap();
    assert_eq!(
        s.tensor_values(next).unwrap(),
        reference.tensor_values(reference_next).unwrap()
    );
}

#[test]
fn checkpoint_nested_replay_bounds_retained_nodes() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(vec![1., 2.], vec![2], true).unwrap();
    let existing = s.tensor_mul(x, x).unwrap();
    let boundary = s.autograd_graph_node_count();
    let out = s
        .checkpoint(&[x], |s, inputs| {
            let mut doubled = inputs[0];
            for _ in 0..8 {
                doubled = s.tensor_add(doubled, doubled)?;
            }
            s.checkpoint(&[doubled], |s, inputs| {
                let square = s.tensor_mul(inputs[0], inputs[0])?;
                s.tensor_sum(square)
            })
        })
        .unwrap();
    assert_eq!(s.autograd_graph_node_count(), boundary + 1);
    assert_eq!(s.tensor_values(existing).unwrap(), [1., 4.]);
    let report = s.tensor_backward(out).unwrap();
    assert_eq!(report.gradient(x), Some(&[131_072., 262_144.][..]));
}

#[test]
fn checkpoint_autocast_matmul_replays_nonuniform_f32_loss() {
    for dtype in [DType::F16, DType::BF16] {
        let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = s
            .tensor_variable_f32(vec![1., 2., 3., 4.], vec![2, 2], true)
            .unwrap();
        let w = s
            .tensor_variable_f32(vec![2., -1., 1., 3.], vec![2, 2], true)
            .unwrap();
        let out = s
            .with_autocast(dtype, |s| {
                s.checkpoint(&[x, w], |s, inputs| s.tensor_matmul(inputs[0], inputs[1]))
            })
            .unwrap();
        assert_eq!(s.tensor_dtype(out).unwrap(), dtype);
        assert_eq!(s.tensor_values_lossy_f64(out).unwrap(), [4., 5., 10., 9.]);
        let weights = s
            .tensor_variable_f32(vec![0.5, -1., 2., 0.25], vec![2, 2], false)
            .unwrap();
        let weighted = s.tensor_mul(out, weights).unwrap();
        assert_eq!(s.tensor_dtype(weighted).unwrap(), DType::F32);
        let loss = s.tensor_sum(weighted).unwrap();
        let report = s.tensor_backward(loss).unwrap();
        assert_eq!(report.gradient(x), Some(&[2., -2.5, 3.75, 2.75][..]));
        assert_eq!(report.gradient(w), Some(&[6.5, -0.25, 9., -1.][..]));
        assert_eq!(s.tensor_dtype(x).unwrap(), DType::F32);
        assert_eq!(s.tensor_dtype(w).unwrap(), DType::F32);
    }
}

fn state_restore_storage_bits(tensor: &ft_core::DenseTensor) -> Vec<u64> {
    use ft_core::TensorStorage;
    match tensor.typed_storage() {
        TensorStorage::F64(values) => values.iter().map(|value| value.to_bits()).collect(),
        TensorStorage::F64Inline4(values) => values.iter().map(|value| value.to_bits()).collect(),
        TensorStorage::F32(values) => values
            .iter()
            .map(|value| u64::from(value.to_bits()))
            .collect(),
        TensorStorage::F16(values) => values
            .iter()
            .map(|value| u64::from(value.to_bits()))
            .collect(),
        TensorStorage::BF16(values) => values
            .iter()
            .map(|value| u64::from(value.to_bits()))
            .collect(),
        TensorStorage::Complex64(values) => values
            .iter()
            .flat_map(|value| [u64::from(value.re.to_bits()), u64::from(value.im.to_bits())])
            .collect(),
        TensorStorage::Complex128(values) => values
            .iter()
            .flat_map(|value| [value.re.to_bits(), value.im.to_bits()])
            .collect(),
        TensorStorage::QInt8(values) => values
            .iter()
            .map(|value| u64::from(value.to_ne_bytes()[0]))
            .collect(),
        TensorStorage::QUInt8(values) => values.iter().map(|&value| u64::from(value)).collect(),
    }
}

#[test]
fn state_restore_preserves_all_storage_variants_and_exact_bits() {
    use ft_core::{
        BFloat16, Complex64, Complex128, DenseTensor, Device, Float16, TensorMeta, TensorStorage,
    };
    use std::sync::Arc;
    let samples = vec![
        TensorStorage::F64(Arc::new(vec![
            1.0,
            -0.0,
            f64::from_bits(0x7ff8_0000_0000_0042),
            f64::INFINITY,
        ])),
        TensorStorage::F64Inline4([
            1.0,
            -0.0,
            f64::from_bits(0x7ff8_0000_0000_0042),
            f64::NEG_INFINITY,
        ]),
        TensorStorage::F32(Arc::new(vec![
            1.0,
            -0.0,
            f32::from_bits(0x7fc0_0042),
            f32::INFINITY,
        ])),
        TensorStorage::F16(Arc::new(vec![
            Float16::from_bits(0x3c00),
            Float16::from_bits(0x8000),
            Float16::from_bits(0x7e42),
            Float16::from_bits(0x7c00),
        ])),
        TensorStorage::BF16(Arc::new(vec![
            BFloat16::from_bits(0x3f80),
            BFloat16::from_bits(0x8000),
            BFloat16::from_bits(0x7fc2),
            BFloat16::from_bits(0x7f80),
        ])),
        TensorStorage::Complex64(Arc::new(vec![
            Complex64::new(1.0, -0.0),
            Complex64::new(f32::from_bits(0x7fc0_0042), 2.0),
            Complex64::new(f32::INFINITY, f32::NEG_INFINITY),
            Complex64::new(-3.0, 4.0),
        ])),
        TensorStorage::Complex128(Arc::new(vec![
            Complex128::new(1.0, -0.0),
            Complex128::new(f64::from_bits(0x7ff8_0000_0000_0042), 2.0),
            Complex128::new(f64::INFINITY, f64::NEG_INFINITY),
            Complex128::new(-3.0, 4.0),
        ])),
        TensorStorage::QInt8(Arc::new(vec![-128, -3, 4, 127])),
        TensorStorage::QUInt8(Arc::new(vec![0, 3, 4, 255])),
    ];
    assert_eq!(samples.len(), 9);
    for storage in samples {
        let dtype = storage.dtype();
        let zeros = match &storage {
            TensorStorage::F64(_) => TensorStorage::F64(Arc::new(vec![0.0; 4])),
            TensorStorage::F64Inline4(_) => TensorStorage::F64Inline4([0.0; 4]),
            TensorStorage::F32(_) => TensorStorage::F32(Arc::new(vec![0.0; 4])),
            TensorStorage::F16(_) => TensorStorage::F16(Arc::new(vec![Float16::ZERO; 4])),
            TensorStorage::BF16(_) => TensorStorage::BF16(Arc::new(vec![BFloat16::ZERO; 4])),
            TensorStorage::Complex64(_) => {
                TensorStorage::Complex64(Arc::new(vec![Complex64::new(0.0, 0.0); 4]))
            }
            TensorStorage::Complex128(_) => {
                TensorStorage::Complex128(Arc::new(vec![Complex128::new(0.0, 0.0); 4]))
            }
            TensorStorage::QInt8(_) => TensorStorage::QInt8(Arc::new(vec![0; 4])),
            TensorStorage::QUInt8(_) => TensorStorage::QUInt8(Arc::new(vec![0; 4])),
        };
        let meta = if dtype.is_quantized() {
            TensorMeta::quantized_from_shape(vec![4], dtype, Device::Cpu, 0.25, 0).unwrap()
        } else {
            TensorMeta::from_shape(vec![4], dtype, Device::Cpu)
        };
        let source = DenseTensor::from_typed_storage(meta.clone(), storage).unwrap();
        let zero = DenseTensor::from_typed_storage(meta.clone(), zeros).unwrap();
        let expected = state_restore_storage_bits(&source);
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let source_node = session.tensor_variable_from_storage(source, false);
        let target_node = session.tensor_variable_from_storage(zero.clone(), false);
        let snapshot = session.tensor_snapshot(source_node).unwrap();
        let before = session.tensor_snapshot(target_node).unwrap();
        let node_count = session.autograd_graph_node_count();
        assert_eq!(snapshot.meta(), &meta);
        assert_ne!(state_restore_storage_bits(&before), expected);
        session
            .tensor_load_state(&[(target_node, snapshot.clone())])
            .unwrap();
        let after = session.tensor_snapshot(target_node).unwrap();
        assert_eq!(state_restore_storage_bits(&after), expected, "{dtype:?}");
        assert_eq!(after.meta(), &meta);
        assert_eq!(after.id(), before.id());
        assert_eq!(after.storage_id(), before.storage_id());
        assert_eq!(after.version(), before.version() + 1);
        assert_eq!(session.autograd_graph_node_count(), node_count);
        assert!(session.is_grad_enabled());
        session.tensor_load_state(&[(source_node, zero)]).unwrap();
        assert_eq!(state_restore_storage_bits(&snapshot), expected);
        assert_eq!(
            state_restore_storage_bits(&session.tensor_snapshot(target_node).unwrap()),
            expected
        );
        assert_eq!(session.autograd_graph_node_count(), node_count);
    }
}

#[test]
fn state_restore_copies_noncontiguous_views_in_logical_order() {
    use ft_core::{DenseTensor, Device, TensorMeta};
    let source_meta =
        TensorMeta::from_shape_and_strides(vec![2, 2], vec![1, 2], 1, DType::F32, Device::Cpu)
            .unwrap();
    let target_meta =
        TensorMeta::from_shape_and_strides(vec![2, 2], vec![3, 1], 1, DType::F32, Device::Cpu)
            .unwrap();
    let source =
        DenseTensor::from_storage_f32(source_meta.clone(), vec![99.0, 1.0, 2.0, 3.0, 4.0, 99.0])
            .unwrap();
    let target = DenseTensor::from_storage_f32(target_meta.clone(), vec![88.0; 8]).unwrap();
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let source_node = session.tensor_variable_from_storage(source, false);
    let target_node = session.tensor_variable_from_storage(target, false);
    let snapshot = session.tensor_snapshot(source_node).unwrap();
    let before = session.tensor_snapshot(target_node).unwrap();
    let node_count = session.autograd_graph_node_count();
    assert_eq!(snapshot.meta(), &source_meta);
    session.no_grad_enter();
    session
        .tensor_load_state(&[(target_node, snapshot)])
        .unwrap();
    assert!(!session.is_grad_enabled());
    session.no_grad_exit();
    assert!(session.is_grad_enabled());
    let restored = session.tensor_snapshot(target_node).unwrap();
    assert_eq!(restored.meta(), &target_meta);
    assert_eq!(restored.id(), before.id());
    assert_eq!(restored.storage_id(), before.storage_id());
    assert_eq!(restored.version(), before.version() + 1);
    assert_eq!(session.autograd_graph_node_count(), node_count);
    assert_eq!(
        restored.typed_storage().as_f32().unwrap(),
        &[88.0, 1.0, 3.0, 88.0, 2.0, 4.0, 88.0, 88.0]
    );
}

#[test]
fn state_restore_late_overlap_error_is_atomic_and_preserves_grad_mode() {
    use ft_autograd::AutogradError;
    use ft_core::{DenseTensor, DenseTensorError, Device, TensorMeta};
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let first = session
        .tensor_variable(vec![7.0, 8.0], vec![2], true)
        .unwrap();
    let overlapping_meta =
        TensorMeta::from_shape_and_strides(vec![2], vec![0], 0, DType::F64, Device::Cpu).unwrap();
    let second = session.tensor_variable_from_storage(
        DenseTensor::from_storage(overlapping_meta, vec![9.0]).unwrap(),
        false,
    );
    let before_first = session.tensor_snapshot(first).unwrap();
    let before_second = session.tensor_snapshot(second).unwrap();
    let updates = vec![
        (
            first,
            DenseTensor::from_contiguous_values(vec![1.0, 2.0], vec![2], Device::Cpu).unwrap(),
        ),
        (
            second,
            DenseTensor::from_contiguous_values(vec![3.0, 4.0], vec![2], Device::Cpu).unwrap(),
        ),
    ];
    let node_count = session.autograd_graph_node_count();
    for grad_enabled in [true, false] {
        if !grad_enabled {
            session.no_grad_enter();
        }
        assert!(matches!(
            session.tensor_load_state(&updates),
            Err(AutogradError::DenseTensor(
                DenseTensorError::OverlappingLayout
            ))
        ));
        assert_eq!(session.tensor_snapshot(first).unwrap(), before_first);
        assert_eq!(session.tensor_snapshot(second).unwrap(), before_second);
        assert_eq!(session.is_grad_enabled(), grad_enabled);
        assert_eq!(session.autograd_graph_node_count(), node_count);
        if !grad_enabled {
            session.no_grad_exit();
        }
    }
    assert!(session.is_grad_enabled());
}

#[test]
fn state_restore_rejects_backward_through_stale_borrowed_forward() {
    use ft_autograd::AutogradError;
    use ft_core::{DenseTensor, Device};
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let parameter = session
        .tensor_variable(vec![2.0, 3.0], vec![2], true)
        .unwrap();
    let version = session.tensor_snapshot(parameter).unwrap().version();
    let square = session.tensor_mul(parameter, parameter).unwrap();
    let loss = session.tensor_sum(square).unwrap();
    let replacement =
        DenseTensor::from_contiguous_values(vec![4.0, 5.0], vec![2], Device::Cpu).unwrap();
    session
        .tensor_load_state(&[(parameter, replacement)])
        .unwrap();
    let result = session.tensor_backward(loss);
    assert!(result.is_err(), "stale forward result: {result:?}");
    assert!(matches!(
        result,
        Err(AutogradError::BorrowedInputVersionMismatch { node, recorded, observed, })
            if node == parameter && recorded == version && observed == version + 1
    ));
}

fn mha_leaf(
    session: &mut FrankenTorchSession,
    values: &[f64],
    shape: &[usize],
    dtype: DType,
    requires_grad: bool,
) -> TensorNodeId {
    if dtype == DType::F64 {
        session
            .tensor_variable(values.to_vec(), shape.to_vec(), requires_grad)
            .unwrap()
    } else {
        assert_eq!(dtype, DType::F32);
        session
            .tensor_variable_f32(
                values.iter().map(|&value| value as f32).collect(),
                shape.to_vec(),
                requires_grad,
            )
            .unwrap()
    }
}

fn mha_assert_close(actual: &[f64], expected: &[f64], dtype: DType) {
    assert_eq!(actual.len(), expected.len());
    let tolerance = if dtype == DType::F64 { 2e-12 } else { 2e-5 };
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance * (1.0 + expected.abs()),
            "entry {index}: actual={actual}, expected={expected}, dtype={dtype:?}"
        );
    }
}

#[test]
fn functional_mha_autocast_preserves_finite_scores_and_full_precision_masks() {
    for dtype in [DType::F16, DType::BF16] {
        for full_precision_mask in [false, true] {
            let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
            let magnitude = if full_precision_mask { 0. } else { 200. };
            let q = mha_leaf(&mut s, &[magnitude; 2], &[1, 1, 2], DType::F32, true);
            let k = mha_leaf(
                &mut s,
                &[magnitude, magnitude, 0., 0.],
                &[1, 2, 2],
                DType::F32,
                true,
            );
            let v = mha_leaf(&mut s, &[2., 4., 6., 8.], &[1, 2, 2], DType::F32, true);
            let w = mha_leaf(
                &mut s,
                &[1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1.],
                &[6, 2],
                DType::F32,
                true,
            );
            let out_w = mha_leaf(&mut s, &[1., 0., 0., 1.], &[2, 2], DType::F32, true);
            let mask = full_precision_mask
                .then(|| mha_leaf(&mut s, &[70_000., 0.], &[1, 2], DType::F32, true));
            // 200*200 + 200*200 overflows F16, but scaling Q first keeps
            // the score finite. In the SDPA branch, the F32 mask must not
            // narrow 70000 to F16 infinity before F32 attention accumulation.
            let need_weights = !full_precision_mask;
            let (output, weights) = s
                .with_autocast(dtype, |s| {
                    s.multi_head_attention_forward(
                        q,
                        k,
                        v,
                        w,
                        None,
                        out_w,
                        None,
                        1,
                        0.,
                        mask,
                        None,
                        need_weights,
                        false,
                        true,
                    )
                })
                .unwrap();
            assert_eq!(s.tensor_dtype(output).unwrap(), dtype);
            assert_eq!(s.tensor_values_lossy_f64(output).unwrap(), [2., 4.]);
            if let Some(weights) = weights {
                assert_eq!(s.tensor_values_lossy_f64(weights).unwrap(), [1., 0.]);
            } else {
                assert!(!need_weights);
            }
            let loss = weighted_loss(&mut s, output, vec![2., -3.]);
            let report = s.tensor_backward(loss).unwrap();
            assert_eq!(s.tensor_gradient(&report, q).unwrap(), [0.; 2]);
            assert_eq!(s.tensor_gradient(&report, k).unwrap(), [0.; 4]);
            assert_eq!(s.tensor_gradient(&report, v).unwrap(), [2., -3., 0., 0.]);
            if let Some(mask) = mask {
                assert_eq!(s.tensor_gradient(&report, mask).unwrap(), [0.; 2]);
            }
        }
    }
}

#[test]
fn functional_mha_additive_masks_and_joint_weight_loss_have_analytic_gradients() {
    // Scores after scaling and both additive masks are ln(2) and ln(6).
    // Thus p=[1/4,3/4]. The returned-weight loss changes the score adjoint
    // from [-3/2,3/2] to [-3/4,3/4], and reaches both floating masks.
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F64, DType::F32] {
            for need_weights in [false, true] {
                let mut s = FrankenTorchSession::new(mode);
                let root_two = 2.0_f64.sqrt();
                let log_three = 3.0_f64.ln();
                let nodes = [
                    mha_leaf(&mut s, &[1., 0.], &[1, 1, 2], dtype, true),
                    mha_leaf(
                        &mut s,
                        &[0., 0., root_two * log_three, 0.],
                        &[1, 2, 2],
                        dtype,
                        true,
                    ),
                    mha_leaf(&mut s, &[2., 0., 6., 0.], &[1, 2, 2], dtype, true),
                    mha_leaf(
                        &mut s,
                        &[1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1.],
                        &[6, 2],
                        dtype,
                        true,
                    ),
                    mha_leaf(&mut s, &[0.; 6], &[6], dtype, true),
                    mha_leaf(&mut s, &[1., 0., 0., 1.], &[2, 2], dtype, true),
                    mha_leaf(&mut s, &[0.; 2], &[2], dtype, true),
                    mha_leaf(&mut s, &[2.0_f64.ln(), 0.], &[1, 2], dtype, true),
                    mha_leaf(&mut s, &[0., 2.0_f64.ln()], &[1, 2], dtype, true),
                ];
                let [q, k, v, w, b, out_w, out_b, mask, padding] = nodes;
                let (output, weights) = s
                    .functional_multi_head_attention_forward(
                        q,
                        k,
                        v,
                        w,
                        Some(b),
                        out_w,
                        Some(out_b),
                        1,
                        0.,
                        Some(mask),
                        Some(padding),
                        need_weights,
                        false,
                        true,
                    )
                    .unwrap();
                assert_eq!(s.tensor_dtype(output).unwrap(), dtype);
                assert_eq!(s.tensor_shape(output).unwrap(), [1, 1, 2]);
                mha_assert_close(
                    &s.tensor_values_lossy_f64(output).unwrap(),
                    &[5., 0.],
                    dtype,
                );
                let mut outputs = vec![output];
                let mut upstream = vec![vec![2., 0.]];
                if let Some(weights) = weights {
                    assert!(need_weights);
                    assert_eq!(s.tensor_dtype(weights).unwrap(), dtype);
                    assert_eq!(s.tensor_shape(weights).unwrap(), [1, 1, 2]);
                    mha_assert_close(
                        &s.tensor_values_lossy_f64(weights).unwrap(),
                        &[0.25, 0.75],
                        dtype,
                    );
                    outputs.push(weights);
                    upstream.push(vec![3., -1.]);
                } else {
                    assert!(!need_weights);
                }
                let gradients = s
                    .tensor_autograd_grad(&outputs, &nodes, Some(&upstream), false, false)
                    .unwrap();
                let g = if need_weights { 0.75 } else { 1.5 };
                let qk = g * log_three;
                let expected = [
                    vec![qk, 0.],
                    vec![-g / root_two, 0., g / root_two, 0.],
                    vec![0.5, 0., 1.5, 0.],
                    vec![qk, 0., 0., 0., qk, 0., 0., 0., 10., 0., 0., 0.],
                    vec![qk, 0., 0., 0., 2., 0.],
                    vec![10., 0., 0., 0.],
                    vec![2., 0.],
                    vec![-g, g],
                    vec![-g, g],
                ];
                for (gradient, expected) in gradients.iter().zip(&expected) {
                    mha_assert_close(gradient.as_deref().unwrap(), expected, dtype);
                }
            }
        }
    }
}

#[test]
fn functional_mha_per_head_masks_preserve_batch_layout_and_weight_gradients() {
    // Four independently chosen head distributions: [3/4,1/4],
    // [1/4,3/4], [2/3,1/3], and [1/2,1/2], ordered by batch then head.
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F64, DType::F32] {
            for batch_first in [false, true] {
                let mut s = FrankenTorchSession::new(mode);
                let nodes = [
                    mha_leaf(&mut s, &[0.; 4], &[2, 1, 2], dtype, true),
                    mha_leaf(&mut s, &[0.; 8], &[2, 2, 2], dtype, true),
                    mha_leaf(
                        &mut s,
                        &[2., 3., 5., 7., 11., 13., 17., 19.],
                        &[2, 2, 2],
                        dtype,
                        true,
                    ),
                    mha_leaf(
                        &mut s,
                        &[1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1.],
                        &[6, 2],
                        dtype,
                        true,
                    ),
                    mha_leaf(&mut s, &[0.; 6], &[6], dtype, true),
                    mha_leaf(&mut s, &[1., 0., 0., 1.], &[2, 2], dtype, true),
                    mha_leaf(&mut s, &[0.; 2], &[2], dtype, true),
                    mha_leaf(
                        &mut s,
                        &[3.0_f64.ln(), 0., 0., 3.0_f64.ln(), 2.0_f64.ln(), 0., 0., 0.],
                        &[4, 1, 2],
                        dtype,
                        true,
                    ),
                ];
                let [q, k, v, w, b, out_w, out_b, mask] = nodes;
                let [q, k, v] = if batch_first {
                    [q, k, v]
                } else {
                    [
                        s.tensor_transpose(q, 0, 1).unwrap(),
                        s.tensor_transpose(k, 0, 1).unwrap(),
                        s.tensor_transpose(v, 0, 1).unwrap(),
                    ]
                };
                let (output, weights) = s
                    .multi_head_attention_forward(
                        q,
                        k,
                        v,
                        w,
                        Some(b),
                        out_w,
                        Some(out_b),
                        2,
                        0.,
                        Some(mask),
                        None,
                        true,
                        false,
                        batch_first,
                    )
                    .unwrap();
                let output = if batch_first {
                    output
                } else {
                    s.tensor_transpose(output, 0, 1).unwrap()
                };
                let weights = weights.unwrap();
                assert_eq!(s.tensor_shape(output).unwrap(), [2, 1, 2]);
                assert_eq!(s.tensor_shape(weights).unwrap(), [2, 1, 2]);
                mha_assert_close(
                    &s.tensor_values_lossy_f64(output).unwrap(),
                    &[2.75, 6., 13., 16.],
                    dtype,
                );
                mha_assert_close(
                    &s.tensor_values_lossy_f64(weights).unwrap(),
                    &[0.5, 0.5, 7. / 12., 5. / 12.],
                    dtype,
                );
                let gradients = s
                    .tensor_autograd_grad(
                        &[output, weights],
                        &nodes,
                        Some(&[vec![2., -1., 3., 4.], vec![1., -1., 2., -2.]]),
                        false,
                        false,
                    )
                    .unwrap();
                let expected = [
                    vec![0.; 4],
                    vec![0.; 8],
                    vec![1.5, -0.25, 0.5, -0.75, 2., 2., 1., 2.],
                    vec![0., 0., 0., 0., 0., 0., 0., 0., 44.5, 53., 51.75, 58.],
                    vec![0., 0., 0., 0., 5., 3.],
                    vec![44.5, 60., 49.25, 58.],
                    vec![5., 3.],
                    vec![
                        -15. / 16.,
                        15. / 16.,
                        15. / 16.,
                        -15. / 16.,
                        -32. / 9.,
                        32. / 9.,
                        -5.5,
                        5.5,
                    ],
                ];
                for (gradient, expected) in gradients.iter().zip(&expected) {
                    mha_assert_close(gradient.as_deref().unwrap(), expected, dtype);
                }
            }
        }
    }
}

#[test]
fn functional_mha_dropout_shares_weights_gradients_and_one_random_draw_sequence() {
    // Independently simulated xoshiro256++ seed 21 begins .8386, .4517,
    // .7179, .4446. At p=1/2 exactly positions 0 and 2 survive, scaled by two.
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F64, DType::F32] {
            for need_weights in [false, true] {
                for probability in [0.5, 1.0] {
                    let mut s = FrankenTorchSession::new(mode);
                    let nodes = [
                        mha_leaf(&mut s, &[1.], &[1, 1, 1], dtype, true),
                        mha_leaf(&mut s, &[0.; 4], &[1, 4, 1], dtype, true),
                        mha_leaf(&mut s, &[2., 4., 6., 8.], &[1, 4, 1], dtype, true),
                        mha_leaf(&mut s, &[1.; 3], &[3, 1], dtype, true),
                        mha_leaf(&mut s, &[1.], &[1, 1], dtype, true),
                    ];
                    let [q, k, v, w, out_w] = nodes;
                    s.manual_seed(21);
                    let mut reference = s.clone();
                    if probability < 1.0 {
                        reference.rand(vec![4], false).unwrap();
                    }
                    let (output, weights) = s
                        .multi_head_attention_forward(
                            q,
                            k,
                            v,
                            w,
                            None,
                            out_w,
                            None,
                            1,
                            probability,
                            None,
                            None,
                            need_weights,
                            false,
                            true,
                        )
                        .unwrap();
                    let value = if probability < 1.0 { 4. } else { 0. };
                    mha_assert_close(&s.tensor_values_lossy_f64(output).unwrap(), &[value], dtype);
                    let mut outputs = vec![output];
                    let mut upstream = vec![vec![2.]];
                    if let Some(weights) = weights {
                        assert!(need_weights);
                        let expected = if probability < 1.0 {
                            vec![0.5, 0., 0.5, 0.]
                        } else {
                            vec![0.; 4]
                        };
                        mha_assert_close(
                            &s.tensor_values_lossy_f64(weights).unwrap(),
                            &expected,
                            dtype,
                        );
                        outputs.push(weights);
                        upstream.push(vec![3., -1., 4., -2.]);
                    } else {
                        assert!(!need_weights);
                    }
                    let gradients = s
                        .tensor_autograd_grad(&outputs, &nodes, Some(&upstream), false, false)
                        .unwrap();
                    let expected = if probability == 1.0 {
                        [vec![0.], vec![0.; 4], vec![0.; 4], vec![0.; 3], vec![0.]]
                    } else {
                        [
                            vec![0.],
                            if need_weights {
                                vec![5. / 8., -23. / 8., 41. / 8., -23. / 8.]
                            } else {
                                vec![0., -2., 4., -2.]
                            },
                            vec![1., 0., 1., 0.],
                            vec![0., 0., 8.],
                            vec![8.],
                        ]
                    };
                    for (gradient, expected) in gradients.iter().zip(&expected) {
                        mha_assert_close(gradient.as_deref().unwrap(), expected, dtype);
                    }
                    let next = s.rand(vec![5], false).unwrap();
                    let expected_next = reference.rand(vec![5], false).unwrap();
                    assert_eq!(
                        s.tensor_values(next).unwrap(),
                        reference.tensor_values(expected_next).unwrap()
                    );
                }
            }
        }
    }
}

#[test]
fn functional_mha_causal_padding_and_fully_masked_rows_follow_each_branch() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F64, DType::F32] {
            for need_weights in [false, true] {
                for case in 0..3 {
                    let mut s = FrankenTorchSession::new(mode);
                    let q = mha_leaf(&mut s, &[0.; 2], &[1, 2, 1], dtype, true);
                    let k = mha_leaf(&mut s, &[0.; 2], &[1, 2, 1], dtype, true);
                    let v = mha_leaf(&mut s, &[2., 6.], &[1, 2, 1], dtype, true);
                    let w = mha_leaf(&mut s, &[1.; 3], &[3, 1], dtype, true);
                    let out_w = mha_leaf(&mut s, &[1.], &[1, 1], dtype, true);
                    let out_b = mha_leaf(&mut s, &[3.], &[1], dtype, true);
                    let mask = match case {
                        0 => Some(mha_leaf(
                            &mut s,
                            &[0., f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
                            &[2, 2],
                            dtype,
                            false,
                        )),
                        1 => None,
                        _ => Some(mha_leaf(&mut s, &[0.; 4], &[2, 2], dtype, false)),
                    };
                    let padding = (case == 2)
                        .then(|| mha_leaf(&mut s, &[0., 3.0_f64.ln()], &[1, 2], dtype, false));
                    let (output, weights) = s
                        .multi_head_attention_forward(
                            q,
                            k,
                            v,
                            w,
                            None,
                            out_w,
                            Some(out_b),
                            1,
                            0.,
                            mask,
                            padding,
                            need_weights,
                            case != 0,
                            true,
                        )
                        .unwrap();
                    let actual = s.tensor_values_lossy_f64(output).unwrap();
                    mha_assert_close(&actual[..1], &[5.], dtype);
                    if case == 0 && need_weights {
                        assert!(actual[1].is_nan());
                    } else {
                        let expected = [3., 7., 8.][case];
                        mha_assert_close(&actual[1..], &[expected], dtype);
                    }
                    if let Some(weights) = weights {
                        let actual = s.tensor_values_lossy_f64(weights).unwrap();
                        mha_assert_close(&actual[..2], &[1., 0.], dtype);
                        if case == 0 {
                            assert!(actual[2..].iter().all(|value| value.is_nan()));
                        } else {
                            let expected = if case == 1 { [0.5, 0.5] } else { [0.25, 0.75] };
                            mha_assert_close(&actual[2..], &expected, dtype);
                        }
                    } else {
                        assert!(!need_weights);
                    }
                }
            }
        }
    }
}

#[test]
fn functional_mha_empty_axes_keep_output_shapes_and_do_not_advance_rng() {
    for dtype in [DType::F64, DType::F32] {
        for (batch, queries, keys) in [(0, 2, 3), (2, 0, 3), (2, 2, 0)] {
            for need_weights in [false, true] {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let q = mha_leaf(
                    &mut s,
                    &vec![0.; batch * queries * 2],
                    &[batch, queries, 2],
                    dtype,
                    true,
                );
                let k = mha_leaf(
                    &mut s,
                    &vec![0.; batch * keys * 2],
                    &[batch, keys, 2],
                    dtype,
                    true,
                );
                let v = mha_leaf(
                    &mut s,
                    &vec![0.; batch * keys * 2],
                    &[batch, keys, 2],
                    dtype,
                    true,
                );
                let w = mha_leaf(
                    &mut s,
                    &[1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1.],
                    &[6, 2],
                    dtype,
                    true,
                );
                let out_w = mha_leaf(&mut s, &[1., 0., 0., 1.], &[2, 2], dtype, true);
                let out_b = mha_leaf(&mut s, &[0.5, -0.5], &[2], dtype, true);
                let mut reference = s.clone();
                let (output, weights) = s
                    .multi_head_attention_forward(
                        q,
                        k,
                        v,
                        w,
                        None,
                        out_w,
                        Some(out_b),
                        1,
                        0.5,
                        None,
                        None,
                        need_weights,
                        true,
                        true,
                    )
                    .unwrap();
                assert_eq!(s.tensor_shape(output).unwrap(), [batch, queries, 2]);
                assert_eq!(s.tensor_dtype(output).unwrap(), dtype);
                let expected: Vec<f64> = (0..batch * queries).flat_map(|_| [0.5, -0.5]).collect();
                assert_eq!(s.tensor_values_lossy_f64(output).unwrap(), expected);
                if let Some(weights) = weights {
                    assert_eq!(s.tensor_shape(weights).unwrap(), [batch, queries, keys]);
                    assert!(s.tensor_values_lossy_f64(weights).unwrap().is_empty());
                } else {
                    assert!(!need_weights);
                }
                let next = s.rand(vec![4], false).unwrap();
                let expected_next = reference.rand(vec![4], false).unwrap();
                assert_eq!(
                    s.tensor_values(next).unwrap(),
                    reference.tensor_values(expected_next).unwrap()
                );
            }
        }
    }
}

#[test]
fn functional_mha_autocast_preserves_f32_parameters_and_exact_weighted_gradients() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F16, DType::BF16] {
            for (need_weights, use_autocast) in
                [(false, false), (false, true), (true, false), (true, true)]
            {
                let mut s = FrankenTorchSession::new(mode);
                let q = mha_leaf(&mut s, &[0.; 2], &[1, 1, 2], DType::F32, true);
                let k = mha_leaf(&mut s, &[0.; 4], &[1, 2, 2], DType::F32, true);
                let v = mha_leaf(&mut s, &[2., 4., 6., 8.], &[1, 2, 2], DType::F32, true);
                let in_w = mha_leaf(
                    &mut s,
                    &[1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1.],
                    &[6, 2],
                    DType::F32,
                    true,
                );
                let in_b = mha_leaf(&mut s, &[0.; 6], &[6], DType::F32, true);
                let out_w = mha_leaf(&mut s, &[1., 0., 0., 1.], &[2, 2], DType::F32, true);
                let out_b = mha_leaf(&mut s, &[0.5, -0.5], &[2], DType::F32, true);
                let mask = mha_leaf(&mut s, &[0., f64::NEG_INFINITY], &[1, 2], DType::F32, false);
                let mut inputs = [q, k, v, in_w, in_b, out_w, out_b];
                if !use_autocast {
                    for input in &mut inputs {
                        *input = s.tensor_to_dtype(*input, dtype).unwrap();
                    }
                }
                let attention = |s: &mut FrankenTorchSession| {
                    s.functional_multi_head_attention_forward(
                        inputs[0],
                        inputs[1],
                        inputs[2],
                        inputs[3],
                        Some(inputs[4]),
                        inputs[5],
                        Some(inputs[6]),
                        1,
                        0.,
                        Some(mask),
                        None,
                        need_weights,
                        false,
                        true,
                    )
                };
                let (output, weights) = if use_autocast {
                    s.with_autocast(dtype, attention)
                } else {
                    attention(&mut s)
                }
                .unwrap();
                assert_eq!(s.tensor_shape(output).unwrap(), [1, 1, 2]);
                assert_eq!(s.tensor_dtype(output).unwrap(), dtype);
                assert!(s.tensor_requires_grad(output).unwrap());
                // The mask selects the first value exactly. Identity projections
                // therefore produce [2, 4] + [0.5, -0.5].
                assert_eq!(s.tensor_values_lossy_f64(output).unwrap(), [2.5, 3.5]);
                if need_weights {
                    let weights = weights.expect("requested attention weights");
                    assert_eq!(s.tensor_shape(weights).unwrap(), [1, 1, 2]);
                    assert_eq!(s.tensor_dtype(weights).unwrap(), dtype);
                    assert_eq!(s.tensor_values_lossy_f64(weights).unwrap(), [1., 0.]);
                    assert!(s.tensor_requires_grad(weights).unwrap());
                } else {
                    assert!(weights.is_none());
                }

                // dL/doutput = [2, -3]. Only the first value row contributes;
                // dL/dW is the outer product of that gradient and [2, 4].
                let loss = weighted_loss(&mut s, output, vec![2., -3.]);
                let report = s.tensor_backward(loss).unwrap();
                for (name, node, expected) in [
                    ("query", q, &[0.; 2][..]),
                    ("key", k, &[0.; 4][..]),
                    ("value", v, &[2., -3., 0., 0.][..]),
                    (
                        "packed projection weight",
                        in_w,
                        &[0., 0., 0., 0., 0., 0., 0., 0., 4., 8., -6., -12.][..],
                    ),
                    (
                        "packed projection bias",
                        in_b,
                        &[0., 0., 0., 0., 2., -3.][..],
                    ),
                    ("output projection weight", out_w, &[4., 8., -6., -12.][..]),
                    ("output projection bias", out_b, &[2., -3.][..]),
                ] {
                    assert_eq!(
                        s.tensor_gradient(&report, node).unwrap(),
                        expected,
                        "{name}: mode={mode:?}, dtype={dtype:?}, need_weights={need_weights}, use_autocast={use_autocast}"
                    );
                }
                for parameter in [q, k, v, in_w, in_b, out_w, out_b] {
                    assert_eq!(s.tensor_dtype(parameter).unwrap(), DType::F32);
                }

                let (inference, inference_weights) = s
                    .with_no_grad(|s| {
                        if use_autocast {
                            s.with_autocast(dtype, attention)
                        } else {
                            attention(s)
                        }
                    })
                    .unwrap();
                assert_eq!(s.tensor_dtype(inference).unwrap(), dtype);
                assert_eq!(s.tensor_shape(inference).unwrap(), [1, 1, 2]);
                assert_eq!(s.tensor_values_lossy_f64(inference).unwrap(), [2.5, 3.5]);
                assert!(!s.tensor_requires_grad(inference).unwrap());
                if need_weights {
                    let weights = inference_weights.expect("requested inference weights");
                    assert_eq!(s.tensor_shape(weights).unwrap(), [1, 1, 2]);
                    assert_eq!(s.tensor_dtype(weights).unwrap(), dtype);
                    assert_eq!(s.tensor_values_lossy_f64(weights).unwrap(), [1., 0.]);
                    assert!(!s.tensor_requires_grad(weights).unwrap());
                } else {
                    assert!(inference_weights.is_none());
                }
                assert!(s.is_grad_enabled());
                assert!(!s.is_autocast_enabled());
            }
        }
    }
}

#[test]
fn functional_mha_preflight_errors_preserve_tape_rng_and_autocast_state() {
    #[derive(Clone, Copy, Debug)]
    enum InvalidInput {
        Shape(usize, &'static [usize]),
        DType(usize),
        Device(usize),
        Missing(usize),
        AttentionMaskShape(&'static [usize]),
        PaddingMaskShape(&'static [usize]),
        AttentionMaskDType,
        PaddingMaskDType,
        AttentionMaskDevice,
        PaddingMaskDevice,
        MissingAttentionMask,
        MissingPaddingMask,
        Dropout(f64),
        Heads(usize),
        ZeroEmbedding,
    }

    fn zeros(s: &mut FrankenTorchSession, shape: &[usize], dtype: DType) -> TensorNodeId {
        mha_leaf(s, &vec![0.; shape.iter().product()], shape, dtype, true)
    }

    fn foreign_device(s: &mut FrankenTorchSession, shape: &[usize]) -> TensorNodeId {
        // This metadata-only fixture must be rejected before backend execution.
        let meta =
            ft_core::TensorMeta::from_shape(shape.to_vec(), DType::F32, ft_core::Device::Cuda);
        let tensor =
            ft_core::DenseTensor::from_storage_f32(meta, vec![0.; shape.iter().product()]).unwrap();
        s.tensor_variable_from_storage(tensor, true)
    }

    fn missing_node(s: &FrankenTorchSession) -> TensorNodeId {
        let mut other = s.clone();
        let absent = other.tensor_variable_f32(vec![0.], vec![1], false).unwrap();
        assert!(s.tensor_shape(absent).is_err(), "fixture ID must be absent");
        absent
    }

    const SHAPES: [&[usize]; 7] = [
        &[1, 1, 2],
        &[1, 2, 2],
        &[1, 2, 2],
        &[6, 2],
        &[6],
        &[2, 2],
        &[2],
    ];
    let mut cases = vec![
        InvalidInput::Shape(0, &[1, 2]),
        InvalidInput::Shape(1, &[2, 2]),
        InvalidInput::Shape(2, &[2, 2]),
        InvalidInput::Shape(0, &[1, 1, 1, 2]),
        InvalidInput::Shape(1, &[2, 2, 2]),
        InvalidInput::Shape(2, &[2, 2, 2]),
        InvalidInput::Shape(1, &[1, 2, 3]),
        InvalidInput::Shape(2, &[1, 2, 3]),
        InvalidInput::Shape(2, &[1, 3, 2]),
        InvalidInput::Shape(3, &[2, 6]),
        InvalidInput::Shape(4, &[2, 3]),
        InvalidInput::Shape(5, &[4, 1]),
        InvalidInput::Shape(6, &[1, 2]),
        InvalidInput::AttentionMaskShape(&[2]),
        InvalidInput::AttentionMaskShape(&[2, 2]),
        InvalidInput::AttentionMaskShape(&[1, 3]),
        InvalidInput::AttentionMaskShape(&[2, 1, 2]),
        InvalidInput::AttentionMaskShape(&[1, 1, 1, 2]),
        InvalidInput::PaddingMaskShape(&[2]),
        InvalidInput::PaddingMaskShape(&[2, 2]),
        InvalidInput::PaddingMaskShape(&[1, 1]),
        InvalidInput::PaddingMaskShape(&[1, 1, 2]),
        InvalidInput::AttentionMaskDType,
        InvalidInput::PaddingMaskDType,
        InvalidInput::AttentionMaskDevice,
        InvalidInput::PaddingMaskDevice,
        InvalidInput::MissingAttentionMask,
        InvalidInput::MissingPaddingMask,
        InvalidInput::Dropout(-0.125),
        InvalidInput::Dropout(1.125),
        InvalidInput::Dropout(f64::NEG_INFINITY),
        InvalidInput::Dropout(f64::INFINITY),
        InvalidInput::Dropout(f64::NAN),
        InvalidInput::Heads(0),
        InvalidInput::Heads(3),
        InvalidInput::ZeroEmbedding,
    ];
    for slot in 0..SHAPES.len() {
        cases.push(InvalidInput::DType(slot));
        cases.push(InvalidInput::Device(slot));
        cases.push(InvalidInput::Missing(slot));
    }

    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for autocast in [None, Some(DType::F16), Some(DType::BF16)] {
            for need_weights in [false, true] {
                for case in cases.iter().copied() {
                    let mut s = FrankenTorchSession::new(mode);
                    s.manual_seed(0x5eed);
                    let mut nodes = SHAPES.map(|shape| zeros(&mut s, shape, DType::F32));
                    nodes[3] = mha_leaf(
                        &mut s,
                        &[1., 0., 0., 1., 1., 0., 0., 1., 1., 0., 0., 1.],
                        &[6, 2],
                        DType::F32,
                        true,
                    );
                    nodes[5] = mha_leaf(&mut s, &[1., 0., 0., 1.], &[2, 2], DType::F32, true);
                    let mut attention_mask = None;
                    let mut padding_mask = None;
                    let mut dropout = 0.25;
                    let mut heads = 1;
                    match case {
                        InvalidInput::Shape(slot, shape) => {
                            nodes[slot] = zeros(&mut s, shape, DType::F32);
                        }
                        InvalidInput::DType(slot) => {
                            nodes[slot] = zeros(&mut s, SHAPES[slot], DType::F64);
                        }
                        InvalidInput::Device(slot) => {
                            nodes[slot] = foreign_device(&mut s, SHAPES[slot]);
                        }
                        InvalidInput::Missing(slot) => nodes[slot] = missing_node(&s),
                        InvalidInput::AttentionMaskShape(shape) => {
                            attention_mask = Some(zeros(&mut s, shape, DType::F32));
                        }
                        InvalidInput::PaddingMaskShape(shape) => {
                            padding_mask = Some(zeros(&mut s, shape, DType::F32));
                        }
                        InvalidInput::AttentionMaskDType => {
                            attention_mask = Some(zeros(&mut s, &[1, 2], DType::F64));
                        }
                        InvalidInput::PaddingMaskDType => {
                            padding_mask = Some(zeros(&mut s, &[1, 2], DType::F64));
                        }
                        InvalidInput::AttentionMaskDevice => {
                            attention_mask = Some(foreign_device(&mut s, &[1, 2]));
                        }
                        InvalidInput::PaddingMaskDevice => {
                            padding_mask = Some(foreign_device(&mut s, &[1, 2]));
                        }
                        InvalidInput::MissingAttentionMask => {
                            attention_mask = Some(missing_node(&s));
                        }
                        InvalidInput::MissingPaddingMask => {
                            padding_mask = Some(missing_node(&s));
                        }
                        InvalidInput::Dropout(value) => dropout = value,
                        InvalidInput::Heads(value) => heads = value,
                        InvalidInput::ZeroEmbedding => {
                            for (slot, shape) in [
                                &[1, 1, 0][..],
                                &[1, 2, 0][..],
                                &[1, 2, 0][..],
                                &[0, 0][..],
                                &[0][..],
                                &[0, 0][..],
                                &[0][..],
                            ]
                            .into_iter()
                            .enumerate()
                            {
                                nodes[slot] = zeros(&mut s, shape, DType::F32);
                            }
                        }
                    }

                    let context = format!(
                        "mode={mode:?}, autocast={autocast:?}, need_weights={need_weights}, case={case:?}"
                    );
                    let before = s.autograd_graph_node_count();
                    let mut reference = s.clone();
                    let forward = |s: &mut FrankenTorchSession| {
                        s.multi_head_attention_forward(
                            nodes[0],
                            nodes[1],
                            nodes[2],
                            nodes[3],
                            Some(nodes[4]),
                            nodes[5],
                            Some(nodes[6]),
                            heads,
                            dropout,
                            attention_mask,
                            padding_mask,
                            need_weights,
                            false,
                            true,
                        )
                    };
                    let result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match autocast {
                            Some(dtype) => s.with_autocast(dtype, forward),
                            None => forward(&mut s),
                        }));
                    assert!(result.is_ok(), "preflight panicked: {context}");
                    assert!(
                        result.unwrap().is_err(),
                        "invalid input accepted: {context}"
                    );
                    assert_eq!(
                        s.autograd_graph_node_count(),
                        before,
                        "tape grew: {context}"
                    );
                    assert!(s.is_grad_enabled(), "grad mode changed: {context}");
                    assert!(!s.is_autocast_enabled(), "autocast leaked: {context}");
                    let actual_next = s.rand(vec![16], false).unwrap();
                    let expected_next = reference.rand(vec![16], false).unwrap();
                    assert_eq!(
                        s.tensor_values(actual_next).unwrap(),
                        reference.tensor_values(expected_next).unwrap(),
                        "RNG advanced: {context}"
                    );
                }
            }
        }
    }
}
