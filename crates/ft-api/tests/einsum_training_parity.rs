//! Independent Cartesian-sum oracle for Einstein contractions and weighted
//! derivatives. Unlike the implementation, this does not lower to matmul,
//! diagonal gathers, pairwise reductions, or broadcasted tensor operations.

use std::collections::BTreeMap;

use ft_api::FrankenTorchSession;
use ft_autograd::TensorNodeId;
use ft_core::{DType, ExecutionMode};

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
