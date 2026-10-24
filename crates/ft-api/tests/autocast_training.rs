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
    let gradients = s
        .tensor_autograd_grad(
            &[out],
            &[x, w, bias],
            Some(&[vec![1., 2., 3., 4., 5., 6.]]),
            false,
            false,
        )
        .unwrap();
    assert_eq!(gradients[0].as_deref(), Some(&[52., 121.][..]));
    assert_eq!(gradients[1].as_deref(), Some(&[14., 19., 24.][..]));
    assert!(gradients[2].is_none());
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
