//! Independent Cartesian-sum oracle for Einstein/bilinear contractions and weighted
//! derivatives. Unlike the implementation, this does not lower to matmul,
//! diagonal gathers, pairwise reductions, or broadcasted tensor operations.

use std::collections::BTreeMap;

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, BackwardOptions, TensorNodeId};
use ft_core::{DType, DenseTensor, Device, ExecutionMode, TensorMeta};

fn variable(
    session: &mut FrankenTorchSession,
    values: &[f64],
    shape: &[usize],
    dtype: DType,
    grad: bool,
) -> TensorNodeId {
    if dtype == DType::F32 {
        session
            .tensor_variable_f32(
                values.iter().map(|&x| x as f32).collect(),
                shape.to_vec(),
                grad,
            )
            .unwrap()
    } else {
        session
            .tensor_variable(values.to_vec(), shape.to_vec(), grad)
            .unwrap()
    }
}

fn values(session: &FrankenTorchSession, tensor: TensorNodeId, dtype: DType) -> Vec<f64> {
    if dtype == DType::F32 {
        session
            .tensor_values_f32(tensor)
            .unwrap()
            .into_iter()
            .map(f64::from)
            .collect()
    } else {
        session.tensor_values(tensor).unwrap()
    }
}

fn close(label: &str, actual: &[f64], expected: &[f64], dtype: DType) {
    assert_eq!(actual.len(), expected.len(), "{label}: length");
    let tolerance = if dtype == DType::F32 { 2e-5 } else { 2e-11 };
    for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (a - e).abs() <= tolerance * (1.0 + e.abs()),
            "{label}[{i}]: {a} != {e}"
        );
    }
}

fn check_case(equation: &str, operands: &[(&str, &[usize])], output_labels: &str) {
    let labels: Vec<Vec<char>> = operands
        .iter()
        .map(|(labels, _)| labels.chars().collect())
        .collect();
    let mut dimensions = BTreeMap::new();
    for ((_, shape), labels) in operands.iter().zip(&labels) {
        for (&label, &size) in labels.iter().zip(*shape) {
            let previous = dimensions.entry(label).or_insert(1);
            if *previous == 1 {
                *previous = size;
            }
        }
    }
    let output_labels: Vec<char> = output_labels.chars().collect();
    let output_shape: Vec<usize> = output_labels
        .iter()
        .map(|label| dimensions[label])
        .collect();
    let output_len: usize = output_shape.iter().product();
    let weights: Vec<f64> = (0..output_len)
        .map(|i| ((i * 3 % 11) as f64 - 4.0) / 4.0)
        .collect();
    let inputs: Vec<Vec<f64>> = operands
        .iter()
        .enumerate()
        .map(|(operand, (_, shape))| {
            (0..shape.iter().product())
                .map(|i| ((i * 5 + operand * 3) % 17) as f64 / 4.0 - 2.0)
                .collect()
        })
        .collect();
    let mut expected = vec![0.0; output_len];
    let mut gradients: Vec<Vec<f64>> = inputs
        .iter()
        .map(|values| vec![0.0; values.len()])
        .collect();
    let axes: Vec<(char, usize)> = dimensions.into_iter().collect();
    let assignments: usize = axes.iter().map(|(_, size)| size).product();
    for assignment in 0..assignments {
        let mut remaining = assignment;
        let mut coordinates = BTreeMap::new();
        for &(label, size) in axes.iter().rev() {
            coordinates.insert(label, remaining % size);
            remaining /= size;
        }
        let output = output_labels
            .iter()
            .zip(&output_shape)
            .fold(0, |flat, (label, &size)| flat * size + coordinates[label]);
        let offsets: Vec<usize> = operands
            .iter()
            .zip(&labels)
            .map(|((_, shape), labels)| {
                labels.iter().zip(*shape).fold(0, |flat, (label, &size)| {
                    flat * size + if size == 1 { 0 } else { coordinates[label] }
                })
            })
            .collect();
        let factors: Vec<f64> = inputs
            .iter()
            .zip(&offsets)
            .map(|(input, &offset)| input[offset])
            .collect();
        expected[output] += factors.iter().product::<f64>();
        for operand in 0..operands.len() {
            let derivative: f64 = factors
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != operand)
                .map(|(_, value)| value)
                .product();
            gradients[operand][offsets[operand]] += weights[output] * derivative;
        }
    }
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F64, DType::F32] {
            for requires_grad in [false, true] {
                let mut session = FrankenTorchSession::new(mode);
                let tensors: Vec<_> = operands
                    .iter()
                    .zip(&inputs)
                    .map(|((_, shape), input)| {
                        variable(&mut session, input, shape, dtype, requires_grad)
                    })
                    .collect();
                let result = session
                    .tensor_einsum(equation, &tensors)
                    .unwrap_or_else(|error| {
                        panic!("{equation} {dtype:?} grad={requires_grad}: {error:?}")
                    });
                assert_eq!(
                    session.tensor_shape(result).unwrap(),
                    output_shape,
                    "{equation}"
                );
                assert_eq!(session.tensor_dtype(result).unwrap(), dtype, "{equation}");
                close(equation, &values(&session, result, dtype), &expected, dtype);
                if requires_grad {
                    let weights = variable(&mut session, &weights, &output_shape, dtype, false);
                    let weighted = session.tensor_mul(result, weights).unwrap();
                    let loss = session.tensor_sum(weighted).unwrap();
                    let report = session.tensor_backward(loss).unwrap();
                    for (&tensor, gradient) in tensors.iter().zip(&gradients) {
                        close(
                            equation,
                            session
                                .tensor_gradient(&report, tensor)
                                .expect("operand gradient"),
                            gradient,
                            dtype,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn contractions_permute_output_coordinates_and_weighted_gradients() {
    check_case("ij,jk->ki", &[("ij", &[2, 3]), ("jk", &[3, 4])], "ki");
    check_case("i,j->ji", &[("i", &[2]), ("j", &[3])], "ji");
    check_case("bi,bj->jbi", &[("bi", &[2, 3]), ("bj", &[2, 4])], "jbi");
    check_case("ab,c->a", &[("ab", &[2, 3]), ("c", &[4])], "a");
    check_case(
        "ab,bc,cd->da",
        &[("ab", &[2, 3]), ("bc", &[3, 4]), ("cd", &[4, 2])],
        "da",
    );
}

#[test]
fn broadcast_contractions_support_singletons_ellipsis_and_uppercase() {
    check_case(
        "bij,bjk->kbi",
        &[("bij", &[1, 2, 3]), ("bjk", &[4, 3, 2])],
        "kbi",
    );
    check_case("i,i->", &[("i", &[1]), ("i", &[5])], "");
    check_case("i,i->", &[("i", &[5]), ("i", &[1])], "");
    check_case(
        " ... A b, ... b C -> C ... A ",
        &[("xyAb", &[2, 1, 2, 3]), ("ybC", &[4, 3, 2])],
        "CxyA",
    );
    check_case("...i,...i->", &[("xyi", &[2, 1, 3]), ("yi", &[4, 3])], "");
    check_case("aZ", &[("aZ", &[2, 3])], "Za");
    check_case(
        "...ij,...jk",
        &[("xij", &[2, 2, 3]), ("jk", &[3, 4])],
        "xik",
    );
}

#[test]
fn repeated_labels_work_in_any_operand_and_preserve_weighted_gradients() {
    check_case("iii->i", &[("iii", &[3, 3, 3])], "i");
    check_case("ijij->ji", &[("ijij", &[2, 3, 2, 3])], "ji");
    check_case("ii,j->ji", &[("ii", &[3, 3]), ("j", &[2])], "ji");
    check_case("iji,jk->ki", &[("iji", &[2, 3, 2]), ("jk", &[3, 4])], "ki");
    check_case(
        "ii,ij,jj->",
        &[("ii", &[2, 2]), ("ij", &[2, 3]), ("jj", &[3, 3])],
        "",
    );
}

#[test]
fn scalar_and_empty_axes_preserve_shape_dtype_and_gradient_edges() {
    check_case("", &[("", &[])], "");
    check_case(",i->i", &[("", &[]), ("i", &[3])], "i");
    check_case("ij,jk->ki", &[("ij", &[2, 0]), ("jk", &[0, 3])], "ki");
    check_case("i,j->ji", &[("i", &[0]), ("j", &[3])], "ji");
    check_case("bi,bj->jbi", &[("bi", &[0, 2]), ("bj", &[1, 3])], "jbi");
    check_case("ii->i", &[("ii", &[0, 0])], "i");
    check_case("i,i->", &[("i", &[1]), ("i", &[0])], "");
}

#[test]
fn diagonal_hessian_and_strided_nonfinite_selection_are_correct() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F64, DType::F32] {
            let mut session = FrankenTorchSession::new(mode);
            let input = variable(
                &mut session,
                &[1., 2., 3., 4., 5., 6., 7., 8.],
                &[2, 2, 2],
                dtype,
                true,
            );
            let diagonal = session.tensor_einsum("iii->", &[input]).unwrap();
            let loss = session.tensor_mul(diagonal, diagonal).unwrap();
            let hessian = session.tensor_functional_hessian(loss, input).unwrap();
            let mut expected = vec![0.0; 64];
            for row in [0, 7] {
                for col in [0, 7] {
                    expected[row * 8 + col] = 2.0;
                }
            }
            close("diagonal Hessian", &hessian, &expected, dtype);

            let input = variable(
                &mut session,
                &[1., f64::NAN, f64::INFINITY, 2.],
                &[2, 2],
                dtype,
                true,
            );
            let transposed = session.tensor_transpose(input, 0, 1).unwrap();
            let result = session.tensor_einsum("ii->i", &[transposed]).unwrap();
            close(
                "strided diagonal",
                &values(&session, result, dtype),
                &[1., 2.],
                dtype,
            );
            let sum = session.tensor_sum(result).unwrap();
            let report = session.tensor_backward(sum).unwrap();
            close(
                "strided diagonal gradient",
                session.tensor_gradient(&report, input).unwrap(),
                &[1., 0., 0., 1.],
                dtype,
            );
        }
    }
}

#[test]
fn malformed_equations_fail_before_creating_graph_nodes() {
    let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = variable(&mut session, &[1.; 4], &[2, 2], DType::F64, true);
    let b = variable(&mut session, &[1.; 6], &[2, 3], DType::F64, true);
    for equation in [
        "ij->ii",
        "ij->z",
        "ij->ij->ij",
        "i-j->ij",
        "i>j->ij",
        "i...j...->ij",
        "i..j->ij",
        "i. . .j->ij",
        "...ij$->...ij",
        "ij->i,j",
    ] {
        let before = variable(&mut session, &[0.], &[], DType::F64, false);
        assert!(session.tensor_einsum(equation, &[a]).is_err(), "{equation}");
        let after = variable(&mut session, &[0.], &[], DType::F64, false);
        assert_eq!(after.0, before.0 + 1, "{equation} mutated the graph");
    }
    assert!(session.tensor_einsum("", &[]).is_err());
    assert!(session.tensor_einsum("ii->i", &[b]).is_err());
    assert!(session.tensor_einsum("ij,kj->ik", &[a, b]).is_err());
}

fn bilinear_call(
    session: &mut FrankenTorchSession,
    inputs: [TensorNodeId; 3],
    bias: Option<TensorNodeId>,
    alias: usize,
) -> Result<TensorNodeId, AutogradError> {
    let [input1, input2, weight] = inputs;
    match alias {
        0 => session.functional_bilinear(input1, input2, weight, bias),
        1 => session.tensor_bilinear(input1, input2, weight, bias),
        _ => session.bilinear_tensor(input1, input2, weight, bias),
    }
}

fn bilinear_loop_oracle(
    inputs: [&[f64]; 4],
    dimensions: [usize; 4],
    upstream: &[f64],
) -> (Vec<f64>, [Vec<f64>; 4]) {
    let [input1, input2, weight, bias] = inputs;
    let [batch, in1, in2, out] = dimensions;
    let mut output = vec![0.; batch * out];
    let mut gradients = inputs.map(|input| vec![0.; input.len()]);
    // Direct scalar definition: each example has its own x1 and x2 row.
    // No tensor contraction, reshape, or implementation helper is the oracle.
    for row in 0..batch {
        for feature in 0..out {
            let output_index = row * out + feature;
            let grad = upstream[output_index];
            output[output_index] = bias[feature];
            gradients[3][feature] += grad;
            for i in 0..in1 {
                for j in 0..in2 {
                    let a = row * in1 + i;
                    let b = row * in2 + j;
                    let w = (feature * in1 + i) * in2 + j;
                    output[output_index] += input1[a] * weight[w] * input2[b];
                    gradients[0][a] += grad * weight[w] * input2[b];
                    gradients[1][b] += grad * input1[a] * weight[w];
                    gradients[2][w] += grad * input1[a] * input2[b];
                }
            }
        }
    }
    (output, gradients)
}

fn bilinear_strided_variable(
    session: &mut FrankenTorchSession,
    logical_values: &[f64],
    shape: &[usize],
    dtype: DType,
    grad: bool,
) -> TensorNodeId {
    // Column-major logical axes, stride-two physical gaps, and a nonzero offset.
    // NaNs outside the logical view expose accidental contiguous-storage reads.
    let mut stride = 2;
    let strides: Vec<usize> = shape
        .iter()
        .map(|&extent| {
            let result = stride;
            stride *= extent;
            result
        })
        .collect();
    let mut physical = vec![f64::NAN; 2 * logical_values.len() + 1];
    for (flat, &value) in logical_values.iter().enumerate() {
        let mut remaining = flat;
        let mut offset = 1;
        for (&extent, &axis_stride) in shape.iter().zip(&strides).rev() {
            offset += (remaining % extent) * axis_stride;
            remaining /= extent;
        }
        physical[offset] = value;
    }
    let physical_len = physical.len();
    let storage = DenseTensor::from_contiguous_values(physical, vec![physical_len], Device::Cpu)
        .unwrap()
        .to_dtype(dtype)
        .unwrap();
    let meta =
        TensorMeta::from_shape_and_strides(shape.to_vec(), strides, 1, dtype, Device::Cpu).unwrap();
    assert!(!meta.is_contiguous());
    session.tensor_variable_from_storage(
        DenseTensor::from_typed_storage(meta, storage.typed_storage().clone()).unwrap(),
        grad,
    )
}

fn check_bilinear_case(batch_shape: &[usize], features: [usize; 3], strided: bool) {
    let [in1, in2, out] = features;
    let batch = batch_shape.iter().product::<usize>();
    let mut input1_shape = batch_shape.to_vec();
    input1_shape.push(in1);
    let mut input2_shape = batch_shape.to_vec();
    input2_shape.push(in2);
    let mut output_shape = batch_shape.to_vec();
    output_shape.push(out);
    let shapes = [input1_shape, input2_shape, vec![out, in1, in2], vec![out]];
    let data: [Vec<f64>; 4] = std::array::from_fn(|operand| {
        (0..shapes[operand].iter().product())
            .map(|i| ((i * 5 + operand * 3) % 17) as f64 / 4.0 - 2.0)
            .collect()
    });
    let upstream: Vec<f64> = (0..batch * out)
        .map(|i| ((i * 3 % 7) as f64 - 3.) / 4.)
        .collect();
    let zero_bias = vec![0.; out];
    for with_bias in [false, true] {
        let (expected, gradients) = bilinear_loop_oracle(
            [
                &data[0],
                &data[1],
                &data[2],
                if with_bias { &data[3] } else { &zero_bias },
            ],
            [batch, in1, in2, out],
            &upstream,
        );
        for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
            for dtype in [DType::F64, DType::F32] {
                for alias in 0..3 {
                    for requires_grad in [false, true] {
                        let mut session = FrankenTorchSession::new(mode);
                        let tensors: [TensorNodeId; 4] = std::array::from_fn(|operand| {
                            if strided {
                                bilinear_strided_variable(
                                    &mut session,
                                    &data[operand],
                                    &shapes[operand],
                                    dtype,
                                    requires_grad,
                                )
                            } else {
                                variable(
                                    &mut session,
                                    &data[operand],
                                    &shapes[operand],
                                    dtype,
                                    requires_grad,
                                )
                            }
                        });
                        let context = format!(
                            "bilinear batch={batch_shape:?} features={features:?} mode={mode:?} dtype={dtype:?} alias={alias} bias={with_bias} grad={requires_grad} strided={strided}"
                        );
                        let result = bilinear_call(
                            &mut session,
                            [tensors[0], tensors[1], tensors[2]],
                            with_bias.then_some(tensors[3]),
                            alias,
                        )
                        .unwrap_or_else(|error| panic!("{context}: {error:?}"));
                        assert_eq!(
                            session.tensor_shape(result).unwrap(),
                            output_shape,
                            "{context}"
                        );
                        assert_eq!(session.tensor_dtype(result).unwrap(), dtype, "{context}");
                        assert_eq!(
                            session.tensor_requires_grad(result).unwrap(),
                            requires_grad,
                            "{context}: output detached"
                        );
                        close(&context, &values(&session, result, dtype), &expected, dtype);
                        if requires_grad {
                            let seed =
                                variable(&mut session, &upstream, &output_shape, dtype, false);
                            let weighted = session.tensor_mul(result, seed).unwrap();
                            let loss = session.tensor_sum(weighted).unwrap();
                            let report = session.tensor_backward(loss).unwrap();
                            let count = if with_bias { 4 } else { 3 };
                            for (operand, (&tensor, expected)) in
                                tensors[..count].iter().zip(&gradients).enumerate()
                            {
                                close(
                                    &format!("{context}: operand {operand}"),
                                    report.gradient(tensor).expect("connected operand gradient"),
                                    expected,
                                    dtype,
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn bilinear_batched_rows_match_scalar_outputs_and_weighted_gradients() {
    check_bilinear_case(&[2], [2, 3, 2], false); // Batch differs from in2.
    check_bilinear_case(&[3], [2, 3, 2], false); // Equal sizes must not mix examples.
    check_bilinear_case(&[3], [1, 1, 1], false); // One scalar product per example.
}

#[test]
fn bilinear_unbatched_and_high_rank_inputs_preserve_the_leading_shape() {
    check_bilinear_case(&[], [2, 3, 2], false);
    check_bilinear_case(&[2, 1, 2], [2, 3, 2], false);
}

#[test]
fn bilinear_noncontiguous_offset_views_use_logical_values_and_gradients() {
    check_bilinear_case(&[2], [2, 3, 2], true);
}

#[test]
fn bilinear_contracts_weight_before_the_second_input_to_avoid_spurious_overflow() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F64, DType::F32] {
            for alias in 0..3 {
                let mut session = FrankenTorchSession::new(mode);
                let exponent = if dtype == DType::F32 { 80 } else { 600 };
                let magnitude = 2.0_f64.powi(exponent);
                let reciprocal = 2.0_f64.powi(-exponent);
                let x = variable(&mut session, &[magnitude], &[1], dtype, true);
                let z = variable(&mut session, &[magnitude], &[1], dtype, true);
                // Exact powers of two make (x * reciprocal) * z = z and
                // (x * 0) * z = 0. Forming x*z first would overflow and
                // incorrectly produce infinity and NaN, respectively.
                // Weight gradients themselves overflow mathematically, so
                // freeze W and check the finite input derivatives instead.
                let w = variable(&mut session, &[reciprocal, 0.], &[2, 1, 1], dtype, false);
                let output = bilinear_call(&mut session, [x, z, w], None, alias).unwrap();
                assert_eq!(session.tensor_shape(output).unwrap(), [2]);
                assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
                assert_eq!(values(&session, output, dtype), [magnitude, 0.]);
                let seed = variable(&mut session, &[1., -2.], &[2], dtype, false);
                let weighted = session.tensor_mul(output, seed).unwrap();
                let loss = session.tensor_sum(weighted).unwrap();
                let report = session.tensor_backward(loss).unwrap();
                assert_eq!(report.gradient(x).unwrap(), [1.]);
                assert_eq!(report.gradient(z).unwrap(), [1.]);
            }
        }
    }
}

#[test]
fn bilinear_empty_batch_or_output_keeps_zero_gradient_dependencies() {
    check_bilinear_case(&[0], [2, 3, 2], false);
    check_bilinear_case(&[2, 0], [2, 3, 2], false);
    check_bilinear_case(&[2], [2, 3, 0], false);
}

#[test]
fn bilinear_product_supports_mixed_second_and_third_derivatives() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F64, DType::F32] {
            for alias in 0..3 {
                let mut session = FrankenTorchSession::new(mode);
                let x = variable(&mut session, &[2.], &[1], dtype, true);
                let z = variable(&mut session, &[3.], &[1], dtype, true);
                let w = variable(&mut session, &[5.], &[1, 1, 1], dtype, true);
                let result = bilinear_call(&mut session, [x, z, w], None, alias).unwrap();
                let weighted = session.tensor_mul_scalar(result, 2.).unwrap();
                let loss = session.tensor_sum(weighted).unwrap();
                // L=2*x*z*w. Its x-adjoint is 2*z*w, its x,z-adjoint
                // is 2*w, and its x,z,w-adjoint is the constant two.
                let mut options = BackwardOptions::for_mode(mode);
                options.create_graph = true;
                options.retain_graph = true;
                let first = session.tensor_backward_with_options(loss, options).unwrap();
                close("dL/dx", first.gradient(x).unwrap(), &[30.], dtype);
                close("dL/dz", first.gradient(z).unwrap(), &[20.], dtype);
                close("dL/dw", first.gradient(w).unwrap(), &[12.], dtype);
                let dx = session.tensor_sum(first.gradient_node(x).unwrap()).unwrap();
                let second = session.tensor_backward_with_options(dx, options).unwrap();
                close("d2L/dxdz", second.gradient(z).unwrap(), &[10.], dtype);
                close("d2L/dxdw", second.gradient(w).unwrap(), &[6.], dtype);
                let dxz = session
                    .tensor_sum(second.gradient_node(z).unwrap())
                    .unwrap();
                let third = session.tensor_backward(dxz).unwrap();
                close("d3L/dxdzdw", third.gradient(w).unwrap(), &[2.], dtype);
            }
        }
    }
}

#[test]
fn bilinear_cpu_autocast_retains_raw_floating_dtype_and_weighted_gradients() {
    let data = [
        vec![1., 2., -1., 3.],
        vec![2., -1., 0.5, 4., 0., -2.],
        vec![1., 0., 2., -1., 3., 0.5, 0.5, -2., 1., 2., 1., -0.5],
        vec![0.5, -1.],
    ];
    let shapes: [&[usize]; 4] = [&[2, 2], &[2, 3], &[2, 2, 3], &[2]];
    let upstream = [1., -2., 3., 4.];
    let (expected, gradients) = bilinear_loop_oracle(
        [&data[0], &data[1], &data[2], &data[3]],
        [2, 2, 3, 2],
        &upstream,
    );
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F64, DType::F32] {
            for autocast in [DType::F16, DType::BF16] {
                let mut session = FrankenTorchSession::new(mode);
                let tensors: [TensorNodeId; 4] = std::array::from_fn(|operand| {
                    variable(&mut session, &data[operand], shapes[operand], dtype, true)
                });
                let result = session
                    .with_autocast(autocast, |s| {
                        s.functional_bilinear(tensors[0], tensors[1], tensors[2], Some(tensors[3]))
                    })
                    .unwrap();
                assert!(!session.is_autocast_enabled());
                assert_eq!(session.tensor_dtype(result).unwrap(), dtype);
                assert_eq!(session.tensor_shape(result).unwrap(), [2, 2]);
                close(
                    "bilinear under CPU autocast",
                    &values(&session, result, dtype),
                    &expected,
                    dtype,
                );
                let seed = variable(&mut session, &upstream, &[2, 2], dtype, false);
                let weighted = session.tensor_mul(result, seed).unwrap();
                let loss = session.tensor_sum(weighted).unwrap();
                let report = session.tensor_backward(loss).unwrap();
                for (&tensor, expected) in tensors.iter().zip(&gradients) {
                    assert_eq!(session.tensor_dtype(tensor).unwrap(), dtype);
                    close(
                        "bilinear CPU autocast gradient",
                        report.gradient(tensor).unwrap(),
                        expected,
                        dtype,
                    );
                }
            }
        }
    }
}

#[test]
fn bilinear_explicit_half_inputs_preserve_dtype_and_original_leaf_gradients() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F16, DType::BF16] {
            let mut session = FrankenTorchSession::new(mode);
            let original = [
                variable(&mut session, &[2., 3.], &[2, 1], DType::F32, true),
                variable(&mut session, &[4., 6., 8., 10.], &[2, 2], DType::F32, true),
                variable(&mut session, &[0.5, -1.], &[1, 1, 2], DType::F32, true),
                variable(&mut session, &[0.25], &[1], DType::F32, true),
            ];
            let inputs = original.map(|input| session.tensor_to_dtype(input, dtype).unwrap());
            let output = session
                .functional_bilinear(inputs[0], inputs[1], inputs[2], Some(inputs[3]))
                .unwrap();
            assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
            assert_eq!(session.tensor_shape(output).unwrap(), [2, 1]);
            assert_eq!(
                session.tensor_values_lossy_f64(output).unwrap(),
                [-7.75, -17.75]
            );
            let output = session.tensor_to_dtype(output, DType::F32).unwrap();
            let seed = variable(&mut session, &[2., -1.], &[2, 1], DType::F32, false);
            let weighted = session.tensor_mul(output, seed).unwrap();
            let loss = session.tensor_sum(weighted).unwrap();
            let report = session.tensor_backward(loss).unwrap();
            let expected = [
                vec![-8., 6.],
                vec![2., -4., -1.5, 3.],
                vec![-8., -6.],
                vec![1.],
            ];
            for (&input, expected) in original.iter().zip(&expected) {
                assert_eq!(session.tensor_dtype(input).unwrap(), DType::F32);
                close(
                    "bilinear explicit-half gradient",
                    report.gradient(input).unwrap(),
                    expected,
                    DType::F32,
                );
            }
            if dtype == DType::F16 {
                // Native bilinear rounds each contraction to the input dtype.
                // 200*400 overflows the first F16 result; multiplying by 0.5
                // must not become finite through whole-operation F32 widening.
                let raw = [
                    variable(&mut session, &[200.], &[1], DType::F32, false),
                    variable(&mut session, &[0.5], &[1], DType::F32, false),
                    variable(&mut session, &[400.], &[1, 1, 1], DType::F32, false),
                ];
                let half = raw.map(|input| session.tensor_to_dtype(input, dtype).unwrap());
                let overflow = session
                    .functional_bilinear(half[0], half[1], half[2], None)
                    .unwrap();
                assert_eq!(session.tensor_dtype(overflow).unwrap(), dtype);
                assert_eq!(
                    session.tensor_values_lossy_f64(overflow).unwrap(),
                    [f64::INFINITY]
                );
            }
        }
    }
}

#[test]
fn bilinear_preflight_rejects_incompatible_inputs_without_mutation() {
    #[derive(Clone, Copy, Debug)]
    enum InvalidInput {
        Shape(usize, &'static [usize]),
        DType(usize),
        Device(usize),
        Missing(usize),
        EmptyFeatures([usize; 2], &'static [usize]),
    }
    const SHAPES: [&[usize]; 4] = [&[2, 2, 2], &[2, 2, 3], &[2, 2, 3], &[2]];
    let mut cases = vec![
        InvalidInput::Shape(0, &[]),
        InvalidInput::Shape(1, &[]),
        InvalidInput::Shape(2, &[]),
        InvalidInput::Shape(2, &[4, 3]),
        InvalidInput::Shape(2, &[1, 2, 2, 3]),
        InvalidInput::Shape(0, &[1, 4, 2]),
        InvalidInput::Shape(1, &[1, 4, 3]),
        InvalidInput::Shape(1, &[4, 3]),
        InvalidInput::Shape(1, &[1, 2, 3]),
        InvalidInput::Shape(0, &[2, 2, 3]),
        InvalidInput::Shape(1, &[2, 2, 2]),
        InvalidInput::Shape(2, &[2, 3, 3]),
        InvalidInput::Shape(2, &[2, 2, 4]),
        InvalidInput::Shape(2, &[0, 2, 3]),
        InvalidInput::Shape(3, &[]),
        InvalidInput::Shape(3, &[1]),
        InvalidInput::Shape(3, &[1, 2]),
        InvalidInput::Shape(3, &[3]),
        // PyTorch's native bilinear flattens each input to [-1, features].
        // Zero features make that reshape ambiguous, even when shapes agree.
        InvalidInput::EmptyFeatures([0, 3], &[2]),
        InvalidInput::EmptyFeatures([2, 0], &[2]),
        InvalidInput::EmptyFeatures([0, 0], &[]),
    ];
    for slot in 0..SHAPES.len() {
        cases.extend([
            InvalidInput::DType(slot),
            InvalidInput::Device(slot),
            InvalidInput::Missing(slot),
        ]);
    }
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for autocast in [None, Some(DType::F16), Some(DType::BF16)] {
            for alias in 0..3 {
                for &case in &cases {
                    let mut session = FrankenTorchSession::new(mode);
                    session.manual_seed(0x5eed);
                    let mut tensors: [TensorNodeId; 4] = std::array::from_fn(|operand| {
                        variable(
                            &mut session,
                            &vec![1.; SHAPES[operand].iter().product()],
                            SHAPES[operand],
                            DType::F32,
                            true,
                        )
                    });
                    match case {
                        InvalidInput::Shape(slot, shape) => {
                            tensors[slot] = variable(
                                &mut session,
                                &vec![1.; shape.iter().product()],
                                shape,
                                DType::F32,
                                true,
                            );
                        }
                        InvalidInput::DType(slot) => {
                            tensors[slot] = variable(
                                &mut session,
                                &vec![1.; SHAPES[slot].iter().product()],
                                SHAPES[slot],
                                DType::F64,
                                true,
                            );
                        }
                        InvalidInput::Device(slot) => {
                            let tensor = DenseTensor::from_contiguous_values_f32(
                                vec![1.; SHAPES[slot].iter().product()],
                                SHAPES[slot].to_vec(),
                                Device::Cuda,
                            )
                            .unwrap();
                            tensors[slot] = session.tensor_variable_from_storage(tensor, true);
                        }
                        InvalidInput::Missing(slot) => {
                            let mut other = session.clone();
                            let absent = variable(&mut other, &[0.], &[], DType::F32, false);
                            assert!(session.tensor_shape(absent).is_err());
                            tensors[slot] = absent;
                        }
                        InvalidInput::EmptyFeatures([in1, in2], batch_shape) => {
                            let input_shape = |features| {
                                let mut shape = batch_shape.to_vec();
                                shape.push(features);
                                shape
                            };
                            let shapes = [
                                input_shape(in1),
                                input_shape(in2),
                                vec![2, in1, in2],
                                vec![2],
                            ];
                            for (slot, shape) in shapes.iter().enumerate() {
                                tensors[slot] = variable(
                                    &mut session,
                                    &vec![1.; shape.iter().product()],
                                    shape,
                                    DType::F32,
                                    true,
                                );
                            }
                        }
                    }
                    let context = format!(
                        "bilinear mode={mode:?} autocast={autocast:?} alias={alias} case={case:?}"
                    );
                    let before = session.autograd_graph_node_count();
                    let mut reference = session.clone();
                    let forward = |s: &mut FrankenTorchSession| {
                        bilinear_call(
                            s,
                            [tensors[0], tensors[1], tensors[2]],
                            Some(tensors[3]),
                            alias,
                        )
                    };
                    let result =
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| match autocast {
                            Some(dtype) => session.with_autocast(dtype, forward),
                            None => forward(&mut session),
                        }));
                    assert!(result.is_ok(), "preflight panicked: {context}");
                    assert!(
                        result.unwrap().is_err(),
                        "invalid input accepted: {context}"
                    );
                    assert_eq!(
                        session.autograd_graph_node_count(),
                        before,
                        "graph grew: {context}"
                    );
                    assert!(!session.is_autocast_enabled(), "autocast leaked: {context}");
                    assert!(session.is_grad_enabled(), "grad state changed: {context}");
                    let next = session.rand(vec![4], false).unwrap();
                    let expected_next = reference.rand(vec![4], false).unwrap();
                    assert_eq!(
                        values(&session, next, DType::F64),
                        values(&reference, expected_next, DType::F64),
                        "RNG advanced: {context}"
                    );
                }
            }
        }
    }
}
