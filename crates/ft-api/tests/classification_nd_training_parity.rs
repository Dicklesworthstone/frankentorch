//! Classification losses with scalar, token, and spatial targets.
//!
//! Forward values and gradients below were independently measured with
//! PyTorch 2.15.0+cpu in float64. The oracle also distinguishes ignored NaN
//! log-probabilities (zero NLL gradient) from ignored NaN logits (NaN CE
//! gradient through log_softmax); those are deliberately different contracts.

use ft_api::FrankenTorchSession;
use ft_autograd::{BackwardOptions, TensorNodeId};
use ft_core::{DType, DenseTensor, Device, ExecutionMode, TensorMeta};

const MODES: [ExecutionMode; 2] = [ExecutionMode::Strict, ExecutionMode::Hardened];
const DTYPES: [DType; 2] = [DType::F64, DType::F32];
const REDUCTIONS: [&str; 3] = ["none", "sum", "mean"];
const SHAPE: [usize; 4] = [1, 3, 2, 2];
const LOGITS: [f64; 12] = [1., -2., 0.5, 3., 0., 1., 2., -1., -1., 0.25, -0.5, 2.];
const PROBABILITIES: [f64; 12] = [
    0.25, 0.5, 0.1, 0.2, 0.5, 0.25, 0.7, 0.3, 0.25, 0.25, 0.2, 0.5,
];
const LOG_PROBS: [f64; 12] = [
    -0.4076059644443804,
    -3.420126188057514,
    -1.7663678998071337,
    -0.3265626412674705,
    -1.4076059644443804,
    -0.4201261880575136,
    -0.26636789980713366,
    -4.32656264126747,
    -2.4076059644443806,
    -1.1701261880575136,
    -2.7663678998071335,
    -1.3265626412674705,
];
const HARD_LOSS: [f64; 4] = [
    0.4076059644443804,
    0.5850630940287568,
    0.5327357996142673,
    0.,
];
const HARD_CE_GRAD: [f64; 12] = [
    -0.3347590442251782,
    0.016354153640417188,
    0.3419055603955805,
    0.,
    0.24472847105479764,
    0.3284819567920835,
    -0.46768558688731554,
    0.,
    0.09003057317038043,
    -0.3448361104325006,
    0.12578002649173503,
    0.,
];

fn tensor(
    session: &mut FrankenTorchSession,
    values: &[f64],
    shape: &[usize],
    dtype: DType,
    requires_grad: bool,
) -> TensorNodeId {
    let storage = DenseTensor::from_contiguous_values(values.to_vec(), shape.to_vec(), Device::Cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    session.tensor_variable_from_storage(storage, requires_grad)
}

fn close(actual: &[f64], expected: &[f64], dtype: DType) {
    assert_eq!(actual.len(), expected.len());
    let tolerance = if dtype == DType::F32 { 8e-6 } else { 2e-10 };
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        if expected.is_nan() {
            assert!(actual.is_nan(), "element {index}: {actual} should be NaN");
        } else {
            assert!(
                (actual - expected).abs() <= tolerance * expected.abs().max(1.0),
                "element {index}: got {actual}, expected {expected}"
            );
        }
    }
}

fn check_reduction(
    session: &FrankenTorchSession,
    output: TensorNodeId,
    events: &[usize],
    unreduced: &[f64],
    denominator: f64,
    reduction: &str,
    dtype: DType,
) {
    let expected = match reduction {
        "none" => unreduced.to_vec(),
        "sum" => vec![unreduced.iter().sum()],
        "mean" => vec![unreduced.iter().sum::<f64>() / denominator],
        _ => unreachable!(),
    };
    let shape = if reduction == "none" { events } else { &[] };
    assert_eq!(session.tensor_shape(output).unwrap(), shape);
    assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
    close(
        &session.tensor_values_lossy_f64(output).unwrap(),
        &expected,
        dtype,
    );
}

fn scaled(values: &[f64], reduction: &str, denominator: f64) -> Vec<f64> {
    let scale = if reduction == "mean" {
        1. / denominator
    } else {
        1.
    };
    values.iter().map(|value| value * scale).collect()
}

#[test]
fn spatial_weighted_hard_ce_and_nll_match_torch_reductions_and_gradients() {
    for mode in MODES {
        for dtype in DTYPES {
            for reduction in REDUCTIONS {
                for nll in [false, true] {
                    let mut session = FrankenTorchSession::new(mode);
                    let input = tensor(
                        &mut session,
                        if nll { &LOG_PROBS } else { &LOGITS },
                        &SHAPE,
                        dtype,
                        true,
                    );
                    let target = tensor(
                        &mut session,
                        &[0., 2., 1., -100.],
                        &[1, 2, 2],
                        DType::F64,
                        false,
                    );
                    let weight = tensor(&mut session, &[1., 2., 0.5], &[3], dtype, false);
                    let output = if nll {
                        session.tensor_nll_loss_full(
                            input,
                            target,
                            Some(weight),
                            Some(-100),
                            reduction,
                        )
                    } else {
                        session.tensor_cross_entropy_full(
                            input,
                            target,
                            Some(weight),
                            Some(-100),
                            reduction,
                        )
                    }
                    .unwrap();
                    check_reduction(
                        &session,
                        output,
                        &[1, 2, 2],
                        &HARD_LOSS,
                        3.5,
                        reduction,
                        dtype,
                    );
                    let root = session.tensor_sum(output).unwrap();
                    let report = session.tensor_backward(root).unwrap();
                    let expected = if nll {
                        [-1., 0., 0., 0., 0., 0., -2., 0., 0., -0.5, 0., 0.]
                    } else {
                        HARD_CE_GRAD
                    };
                    close(
                        report.gradient(input).unwrap(),
                        &scaled(&expected, reduction, 3.5),
                        dtype,
                    );
                }
            }
        }
    }
}

#[test]
fn unbatched_hard_and_soft_targets_have_scalar_losses() {
    for mode in MODES {
        for dtype in DTYPES {
            for reduction in REDUCTIONS {
                for route in 0..4 {
                    let mut session = FrankenTorchSession::new(mode);
                    let nll = route == 1;
                    let soft = route == 3;
                    let input_values = if nll {
                        [
                            -0.5045969023422835,
                            -3.5045969023422834,
                            -1.0045969023422834,
                        ]
                    } else {
                        [1., -2., 0.5]
                    };
                    let input = tensor(&mut session, &input_values, &[3], dtype, true);
                    let target = if soft {
                        tensor(&mut session, &[0.2, 0.3, 0.5], &[3], dtype, true)
                    } else {
                        tensor(&mut session, &[2.], &[], DType::F64, false)
                    };
                    let weight = tensor(&mut session, &[1., 2., 0.5], &[3], dtype, false);
                    let output = match route {
                        0 => session.tensor_cross_entropy_full(
                            input,
                            target,
                            Some(weight),
                            None,
                            reduction,
                        ),
                        1 => session.tensor_nll_loss_full(
                            input,
                            target,
                            Some(weight),
                            None,
                            reduction,
                        ),
                        _ => session.tensor_cross_entropy(input, target, reduction),
                    }
                    .unwrap();
                    let (loss, gradient, denominator) = match route {
                        0 => (
                            0.5022984511711417,
                            [
                                0.3018744480743129,
                                0.015029443784787011,
                                -0.3169038918590999,
                            ],
                            0.5,
                        ),
                        1 => (0.5022984511711417, [0., 0., -0.5], 0.5),
                        2 => (
                            1.0045969023422834,
                            [
                                0.6037488961486258,
                                0.030058887569574022,
                                -0.6338077837181998,
                            ],
                            1.,
                        ),
                        _ => (
                            1.6545969023422833,
                            [
                                0.4037488961486258,
                                -0.26994111243042596,
                                -0.1338077837181998,
                            ],
                            1.,
                        ),
                    };
                    check_reduction(
                        &session,
                        output,
                        &[],
                        &[loss],
                        denominator,
                        reduction,
                        dtype,
                    );
                    let report = session.tensor_backward(output).unwrap();
                    close(
                        report.gradient(input).unwrap(),
                        &scaled(&gradient, reduction, denominator),
                        dtype,
                    );
                    if soft {
                        close(
                            report.gradient(target).unwrap(),
                            &[0.5045969023422835, 3.5045969023422834, 1.0045969023422834],
                            dtype,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn token_batches_keep_the_class_axis_between_batch_and_sequence() {
    let losses = [
        0.6802696706417346,
        2.132845233727575,
        3.1238729583693674,
        3.201765469284433,
    ];
    let gradient = [
        -0.493519608944346,
        0.005899750401902781,
        0.3071958857184984,
        0.8756005950630878,
        0.18632372322584756,
        -0.8815003454649903,
        0.8834920740271945,
        -0.9593096969526053,
        -0.9560135197079421,
        0.14202311267846235,
        0.07252144568074775,
        0.8172865842741428,
    ];
    for mode in MODES {
        for dtype in DTYPES {
            for reduction in REDUCTIONS {
                let mut session = FrankenTorchSession::new(mode);
                let input = tensor(&mut session, &LOGITS, &[2, 3, 2], dtype, true);
                let target = tensor(&mut session, &[0., 2., 1., 0.], &[2, 2], DType::F64, false);
                let output = session
                    .functional_cross_entropy(input, target, reduction)
                    .unwrap();
                check_reduction(&session, output, &[2, 2], &losses, 4., reduction, dtype);
                let root = session.tensor_sum(output).unwrap();
                let report = session.tensor_backward(root).unwrap();
                close(
                    report.gradient(input).unwrap(),
                    &scaled(&gradient, reduction, 4.),
                    dtype,
                );
            }
        }
    }
}

#[test]
fn spatial_soft_weights_and_smoothing_preserve_target_gradients() {
    let losses = [
        [
            1.4076059644443806,
            2.107626188057514,
            0.9163678998071336,
            2.0265626412674704,
        ],
        [
            1.810458201111023,
            2.066391961564703,
            0.8261886396914138,
            2.9928907733308434,
        ],
        [
            1.4076059644443804,
            2.020126188057514,
            1.0530345664738003,
            2.0198959746008036,
        ],
    ];
    let gradients = [
        [
            0.4152409557748218,
            -0.4672916927191656,
            0.07095278019779025,
            0.5213991842739687,
            -0.25527152894520233,
            0.40696391358416695,
            0.06615720655634227,
            -0.2867871130462106,
            -0.15996942682961957,
            0.060327779134998816,
            -0.1371099867541325,
            -0.23461207122775807,
        ],
        [
            0.66470631419038,
            -0.46320315430906134,
            0.1735244483164644,
            0.5574691434876671,
            -0.6634983522996533,
            0.23908440278218782,
            -0.17414846950985227,
            -0.5861264686985211,
            -0.0012079618907269024,
            0.22411875152687366,
            0.0006240211933880247,
            0.02865732521085404,
        ],
        [
            0.3985742891081551,
            -0.43395835938583227,
            0.02428611353112357,
            0.49473251760730197,
            -0.22193819561186903,
            0.3902972469175003,
            0.13949053988967555,
            -0.2934537797128772,
            -0.17663609349628623,
            0.04366111246833215,
            -0.16377665342079917,
            -0.20127873789442474,
        ],
    ];
    for mode in MODES {
        for dtype in DTYPES {
            for reduction in REDUCTIONS {
                for route in 0..4 {
                    let kind = match route {
                        0 | 1 => 0,
                        2 => 1,
                        _ => 2,
                    };
                    let mut session = FrankenTorchSession::new(mode);
                    let input = tensor(&mut session, &LOGITS, &SHAPE, dtype, true);
                    let target = tensor(&mut session, &PROBABILITIES, &SHAPE, dtype, true);
                    let weight = tensor(&mut session, &[1., 2., 0.5], &[3], dtype, false);
                    let output = match route {
                        0 => session.tensor_cross_entropy_soft(input, target, reduction),
                        1 => session.tensor_cross_entropy(input, target, reduction),
                        2 => session.tensor_cross_entropy_full(
                            input,
                            target,
                            Some(weight),
                            None,
                            reduction,
                        ),
                        _ => session
                            .tensor_cross_entropy_label_smoothing(input, target, 0.2, reduction),
                    }
                    .unwrap();
                    // Soft means divide by event count, NOT weighted hard-target mass.
                    check_reduction(
                        &session,
                        output,
                        &[1, 2, 2],
                        &losses[kind],
                        4.,
                        reduction,
                        dtype,
                    );
                    let root = session.tensor_sum(output).unwrap();
                    let report = session.tensor_backward(root).unwrap();
                    close(
                        report.gradient(input).unwrap(),
                        &scaled(&gradients[kind], reduction, 4.),
                        dtype,
                    );
                    let target_gradient: Vec<_> = LOG_PROBS
                        .iter()
                        .enumerate()
                        .map(|(index, &lp)| {
                            let factor = match kind {
                                1 => [1., 2., 0.5][index / 4],
                                2 => 0.8,
                                _ => 1.,
                            };
                            -lp * factor
                        })
                        .collect();
                    close(
                        report.gradient(target).unwrap(),
                        &scaled(&target_gradient, reduction, 4.),
                        dtype,
                    );
                }
            }
        }
    }
}

#[test]
fn spatial_hard_label_smoothing_matches_torch() {
    let losses = [
        0.6076059644443805,
        1.2701261880575136,
        0.5330345664738003,
        1.4598959746008038,
    ];
    let gradient = [
        -0.2014257108918449,
        -0.03395835938583229,
        0.10428611353112359,
        0.654732517607302,
        0.17806180438813096,
        0.5902972469175003,
        -0.10050946011032447,
        -0.05345377971287725,
        0.02336390650371377,
        -0.5563388875316679,
        -0.0037766534207991503,
        -0.6012787378944248,
    ];
    for mode in MODES {
        for dtype in DTYPES {
            for reduction in REDUCTIONS {
                let mut session = FrankenTorchSession::new(mode);
                let input = tensor(&mut session, &LOGITS, &SHAPE, dtype, true);
                let target = tensor(
                    &mut session,
                    &[0., 2., 1., 2.],
                    &[1, 2, 2],
                    DType::F64,
                    false,
                );
                let output = session
                    .tensor_cross_entropy_label_smoothing(input, target, 0.2, reduction)
                    .unwrap();
                check_reduction(&session, output, &[1, 2, 2], &losses, 4., reduction, dtype);
                let root = session.tensor_sum(output).unwrap();
                let report = session.tensor_backward(root).unwrap();
                close(
                    report.gradient(input).unwrap(),
                    &scaled(&gradient, reduction, 4.),
                    dtype,
                );
            }
        }
    }
}

#[test]
fn ignored_nan_logprobabilities_and_logits_follow_distinct_torch_gradients() {
    for mode in MODES {
        for dtype in DTYPES {
            for reduction in REDUCTIONS {
                for nll in [false, true] {
                    for all_ignored in [false, true] {
                        for ignored_nan in [false, true] {
                            let mut session = FrankenTorchSession::new(mode);
                            let first = if ignored_nan { f64::NAN } else { 1. };
                            let input = tensor(
                                &mut session,
                                &[first, 4., first, 5., first, 6.],
                                &[1, 3, 2],
                                dtype,
                                true,
                            );
                            let target = tensor(
                                &mut session,
                                &[-100., if all_ignored { -100. } else { 1. }],
                                &[1, 2],
                                DType::F64,
                                false,
                            );
                            let weight =
                                tensor(&mut session, &[f64::NAN, 2., 3.], &[3], dtype, false);
                            let output = if nll {
                                session.tensor_nll_loss_full(
                                    input,
                                    target,
                                    Some(weight),
                                    Some(-100),
                                    reduction,
                                )
                            } else {
                                session.tensor_cross_entropy_full(
                                    input,
                                    target,
                                    Some(weight),
                                    Some(-100),
                                    reduction,
                                )
                            }
                            .unwrap();
                            let last = if all_ignored {
                                0.
                            } else if nll {
                                -10.
                            } else {
                                2.8152119288887607
                            };
                            check_reduction(
                                &session,
                                output,
                                &[1, 2],
                                &[0., last],
                                if all_ignored { 0. } else { 2. },
                                reduction,
                                dtype,
                            );
                            let root = session.tensor_sum(output).unwrap();
                            let report = session.tensor_backward(root).unwrap();
                            let mut gradient = if all_ignored {
                                vec![0.; 6]
                            } else if nll {
                                scaled(&[0., 0., 0., -2., 0., 0.], reduction, 2.)
                            } else {
                                scaled(
                                    &[
                                        0.,
                                        0.18006114634076087,
                                        0.,
                                        -1.5105430578904047,
                                        0.,
                                        1.3304819115496436,
                                    ],
                                    reduction,
                                    2.,
                                )
                            };
                            if ignored_nan && !nll {
                                for index in [0, 2, 4] {
                                    gradient[index] = f64::NAN;
                                }
                            }
                            close(report.gradient(input).unwrap(), &gradient, dtype);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn empty_spatial_batches_keep_empty_gradients_for_every_reduction() {
    for mode in MODES {
        for dtype in DTYPES {
            for (shape, events) in [(vec![0, 3, 2], vec![0, 2]), (vec![2, 3, 0], vec![2, 0])] {
                for reduction in REDUCTIONS {
                    for nll in [false, true] {
                        let mut session = FrankenTorchSession::new(mode);
                        let input = tensor(&mut session, &[], &shape, dtype, true);
                        let target = tensor(&mut session, &[], &events, DType::F64, false);
                        let output = if nll {
                            session.tensor_nll_loss(input, target, reduction)
                        } else {
                            session.tensor_cross_entropy(input, target, reduction)
                        }
                        .unwrap();
                        check_reduction(&session, output, &events, &[], 0., reduction, dtype);
                        let root = session.tensor_sum(output).unwrap();
                        let report = session.tensor_backward(root).unwrap();
                        assert!(report.gradient(input).unwrap().is_empty());
                    }
                }
            }
        }
    }
}

#[test]
fn spatial_classification_reads_logical_strides_and_preserves_original_metadata() {
    for mode in MODES {
        for dtype in DTYPES {
            let mut session = FrankenTorchSession::new(mode);
            let storage = DenseTensor::from_contiguous_values(
                vec![
                    99., 99., 1., 0., -1., -2., 1., 0.25, 0.5, 2., -0.5, 3., -1., 2., 99.,
                ],
                vec![15],
                Device::Cpu,
            )
            .unwrap()
            .to_dtype(dtype)
            .unwrap();
            let meta = TensorMeta::from_shape_and_strides(
                SHAPE.to_vec(),
                vec![12, 1, 6, 3],
                2,
                dtype,
                Device::Cpu,
            )
            .unwrap();
            let input = session.tensor_variable_from_storage(
                DenseTensor::from_typed_storage(meta.clone(), storage.typed_storage().clone())
                    .unwrap(),
                true,
            );
            let target_storage = DenseTensor::from_contiguous_values(
                vec![99., 0., 99., 2., 99., 1., 99., -100.],
                vec![8],
                Device::Cpu,
            )
            .unwrap();
            let target_meta = TensorMeta::from_shape_and_strides(
                vec![1, 2, 2],
                vec![8, 4, 2],
                1,
                DType::F64,
                Device::Cpu,
            )
            .unwrap();
            let target = session.tensor_variable_from_storage(
                DenseTensor::from_typed_storage(
                    target_meta,
                    target_storage.typed_storage().clone(),
                )
                .unwrap(),
                false,
            );
            let weight = tensor(&mut session, &[1., 2., 0.5], &[3], dtype, false);
            let output = session
                .tensor_cross_entropy_full(input, target, Some(weight), Some(-100), "mean")
                .unwrap();
            check_reduction(&session, output, &[1, 2, 2], &HARD_LOSS, 3.5, "mean", dtype);
            let report = session.tensor_backward(output).unwrap();
            close(
                report.gradient(input).unwrap(),
                &scaled(&HARD_CE_GRAD, "mean", 3.5),
                dtype,
            );
            assert_eq!(session.tensor_snapshot(input).unwrap().meta(), &meta);
        }
    }
}

#[test]
fn invalid_class_targets_and_options_fail_before_graph_mutation() {
    for mode in MODES {
        for scalar in [false, true] {
            let mut session = FrankenTorchSession::new(mode);
            let (shape, events): (&[usize], &[usize]) = if scalar {
                (&[3], &[])
            } else {
                (&SHAPE, &[1, 2, 2])
            };
            let input = tensor(
                &mut session,
                if scalar { &[1., 2., 3.] } else { &LOGITS },
                shape,
                DType::F64,
                true,
            );
            for invalid in [
                0.5,
                -0.5,
                -100.5,
                -101.,
                3.,
                f64::NAN,
                f64::INFINITY,
                f64::NEG_INFINITY,
            ] {
                let mut values = vec![0.; events.iter().product()];
                values[0] = invalid;
                let target = tensor(&mut session, &values, events, DType::F64, false);
                let before = session.autograd_graph_node_count();
                assert!(session.tensor_cross_entropy(input, target, "mean").is_err());
                assert_eq!(session.autograd_graph_node_count(), before);
                assert!(
                    session
                        .tensor_cross_entropy_full(input, target, None, Some(-100), "mean")
                        .is_err()
                );
                assert_eq!(session.autograd_graph_node_count(), before);
                assert!(session.tensor_nll_loss(input, target, "mean").is_err());
                assert_eq!(session.autograd_graph_node_count(), before);
                assert!(
                    session
                        .tensor_nll_loss_full(input, target, None, Some(-100), "mean")
                        .is_err()
                );
                assert_eq!(session.autograd_graph_node_count(), before);
                assert!(
                    session
                        .tensor_cross_entropy_label_smoothing(input, target, 0.2, "mean")
                        .is_err()
                );
                assert_eq!(session.autograd_graph_node_count(), before);
            }
            let valid = tensor(
                &mut session,
                &vec![0.; events.iter().product()],
                events,
                DType::F64,
                false,
            );
            let before = session.autograd_graph_node_count();
            assert!(
                session
                    .tensor_cross_entropy(input, valid, "invalid")
                    .is_err()
            );
            assert!(session.tensor_nll_loss(input, valid, "invalid").is_err());
            assert!(
                session
                    .tensor_cross_entropy_label_smoothing(input, valid, f64::NAN, "mean")
                    .is_err()
            );
            assert!(
                session
                    .tensor_cross_entropy_label_smoothing(input, valid, 1.1, "mean")
                    .is_err()
            );
            assert_eq!(session.autograd_graph_node_count(), before);
        }
    }
}

#[test]
fn unbatched_cross_entropy_hessian_vector_matches_torch() {
    for mode in MODES {
        for dtype in DTYPES {
            let mut session = FrankenTorchSession::new(mode);
            let input = tensor(&mut session, &[0.25, -0.5, 1.25], &[3], dtype, true);
            let target = tensor(&mut session, &[2.], &[], DType::F64, false);
            let output = session
                .tensor_cross_entropy_full(input, target, None, None, "mean")
                .unwrap();
            close(
                &session.tensor_values_lossy_f64(output).unwrap(),
                &[0.43285546687876886],
                dtype,
            );
            let first = session
                .tensor_backward_with_options(
                    output,
                    BackwardOptions::for_mode(mode).with_create_graph(true),
                )
                .unwrap();
            close(
                first.gradient(input).unwrap(),
                &[
                    0.23862655824004822,
                    0.11271920470830456,
                    -0.35134576294835285,
                ],
                dtype,
            );
            let gradient = first.gradient_node(input).unwrap();
            let direction = tensor(&mut session, &[1., -2., 0.5], &[3], DType::F64, false);
            let weighted = session.tensor_mul(gradient, direction).unwrap();
            let scalar = session.tensor_sum(weighted).unwrap();
            let second = session.tensor_backward(scalar).unwrap();
            close(
                second.gradient(input).unwrap(),
                &[0.15808645163902393, -0.2634828619291288, 0.1053964102901049],
                dtype,
            );
        }
    }
}

#[test]
fn zero_class_losses_preserve_torch_nan_values_and_empty_gradient_edges() {
    // PyTorch 2.15: probability-target CE over zero classes has zero
    // unreduced/summed loss and NaN mean. Hard-target label smoothing divides
    // by the zero class count, including in the empty-batch summed reduction.
    for mode in MODES {
        for dtype in DTYPES {
            for samples in [2, 0] {
                for reduction in REDUCTIONS {
                    for soft in [false, true] {
                        let mut session = FrankenTorchSession::new(mode);
                        let input = tensor(&mut session, &[], &[samples, 0], dtype, true);
                        let target = if soft {
                            tensor(&mut session, &[], &[samples, 0], dtype, true)
                        } else {
                            tensor(
                                &mut session,
                                &vec![-100.; samples],
                                &[samples],
                                DType::F64,
                                false,
                            )
                        };
                        let output = if soft {
                            session.tensor_cross_entropy_soft(input, target, reduction)
                        } else {
                            session
                                .tensor_cross_entropy_label_smoothing(input, target, 0.2, reduction)
                        }
                        .unwrap();
                        let expected = match (soft, reduction) {
                            (true, "none") => vec![0.; samples],
                            (true, "sum") => vec![0.],
                            (false, "none") => vec![f64::NAN; samples],
                            _ => vec![f64::NAN],
                        };
                        let expected_shape = if reduction == "none" {
                            vec![samples]
                        } else {
                            vec![]
                        };
                        assert_eq!(session.tensor_shape(output).unwrap(), expected_shape);
                        assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
                        close(
                            &session.tensor_values_lossy_f64(output).unwrap(),
                            &expected,
                            dtype,
                        );
                        let root = session.tensor_sum(output).unwrap();
                        let report = session.tensor_backward(root).unwrap();
                        assert!(report.gradient(input).unwrap().is_empty());
                        if soft {
                            assert!(report.gradient(target).unwrap().is_empty());
                        }
                    }
                }
            }
        }
    }
}
