//! Regression: complex / FFT / IFFT autograd backward parity vs PyTorch 2.12.
//!
//! frankentorch-ng1hw closed the "last differentiable gaps" — the real<->complex
//! bridge ops and the FFT family backward. That surface was only exercised by the
//! `complex_autograd_probe` example (manual eyeball vs torch), with NO CI test and
//! a stale golden comment on the ifft(n=4) chain. This locks the verified torch
//! goldens so a regression (or a future "fix" toward a wrong golden) fails loudly.
//!
//! Goldens generated with torch 2.12.0+cpu (f64), e.g. for chain G:
//!   xr=[1,2,3,4]; xi=[.5,1,1.5,2]; y=ifft(complex(xr,xi), n=4)
//!   loss=sum(real(y)*[.1,.2,.3,.4] + imag(y)*[.5,1,1.5,2]); loss.backward()
//!   -> xr.grad=[0.25,-0.3,-0.05,0.2]; xi.grad=[1.25,-0.2,-0.25,-0.3]

use ft_api::FrankenTorchSession;
use ft_autograd::{AutogradError, BackwardOptions, TensorNodeId};
use ft_core::{DType, ExecutionMode};

fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!((g - w).abs() <= 1e-9, "{label}[{i}]: got {g}, want {w}");
    }
}

fn s() -> FrankenTorchSession {
    FrankenTorchSession::new(ExecutionMode::Strict)
}

/// Chain A: real/imag bridge — loss = sum(real(z)*2 + imag(z)*3).
#[test]
fn complex_real_imag_bridge_backward_matches_torch() {
    let mut s = s();
    let re = s
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
        .unwrap();
    let im = s
        .tensor_variable(vec![0.5, 1.5, 2.5, 3.5], vec![4], true)
        .unwrap();
    let z = s.tensor_complex(re, im).unwrap();
    let rp = s.tensor_real(z).unwrap();
    let ip = s.tensor_imag(z).unwrap();
    let a = s.tensor_mul_scalar(rp, 2.0).unwrap();
    let b = s.tensor_mul_scalar(ip, 3.0).unwrap();
    let sum = s.tensor_add(a, b).unwrap();
    let out = s.tensor_sum(sum).unwrap();
    let rep = s.tensor_backward(out).unwrap();
    assert_close(
        "A re",
        s.tensor_gradient(&rep, re).unwrap(),
        &[2.0, 2.0, 2.0, 2.0],
    );
    assert_close(
        "A im",
        s.tensor_gradient(&rep, im).unwrap(),
        &[3.0, 3.0, 3.0, 3.0],
    );
}

/// Chain B: abs(z) — grad re = re/|z|, im = im/|z|.
#[test]
fn complex_abs_backward_matches_torch() {
    let mut s = s();
    let re = s
        .tensor_variable(vec![3.0, 1.0, -2.0, 0.5], vec![4], true)
        .unwrap();
    let im = s
        .tensor_variable(vec![4.0, 1.0, 2.0, -0.5], vec![4], true)
        .unwrap();
    let z = s.tensor_complex(re, im).unwrap();
    let mag = s.tensor_abs(z).unwrap();
    let out = s.tensor_sum(mag).unwrap();
    let rep = s.tensor_backward(out).unwrap();
    assert_close(
        "B re",
        s.tensor_gradient(&rep, re).unwrap(),
        &[
            0.6,
            std::f64::consts::FRAC_1_SQRT_2,
            -std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
        ],
    );
    assert_close(
        "B im",
        s.tensor_gradient(&rep, im).unwrap(),
        &[
            0.8,
            std::f64::consts::FRAC_1_SQRT_2,
            std::f64::consts::FRAC_1_SQRT_2,
            -std::f64::consts::FRAC_1_SQRT_2,
        ],
    );
}

/// Chain D: real -> fft (no resize) -> weighted real/imag loss.
#[test]
fn fft_backward_matches_torch() {
    let mut s = s();
    let x = s
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
        .unwrap();
    let y = s.tensor_fft(x, None).unwrap();
    let re = s.tensor_real(y).unwrap();
    let im = s.tensor_imag(y).unwrap();
    let a = s
        .tensor_variable(vec![0.1, 0.2, 0.3, 0.4], vec![4], false)
        .unwrap();
    let b = s
        .tensor_variable(vec![0.5, 0.6, 0.7, 0.8], vec![4], false)
        .unwrap();
    let ra = s.tensor_mul(re, a).unwrap();
    let ib = s.tensor_mul(im, b).unwrap();
    let sum = s.tensor_add(ra, ib).unwrap();
    let out = s.tensor_sum(sum).unwrap();
    let rep = s.tensor_backward(out).unwrap();
    assert_close(
        "D fft",
        s.tensor_gradient(&rep, x).unwrap(),
        &[1.0, 0.0, -0.2, -0.4],
    );
}

/// Chains E/F: fft with zero-padding (n=6) and truncation (n=2).
#[test]
fn fft_resize_backward_matches_torch() {
    for (label, n, golden) in [
        (
            "E pad n=6",
            6usize,
            vec![2.1, 2.298076211, 0.566025404, -0.3],
        ),
        ("F trunc n=2", 2usize, vec![0.3, -0.1, 0.0, 0.0]),
    ] {
        let mut s = s();
        let x = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
            .unwrap();
        let y = s.tensor_fft(x, Some(n)).unwrap();
        let re = s.tensor_real(y).unwrap();
        let im = s.tensor_imag(y).unwrap();
        let av: Vec<f64> = (1..=n).map(|k| k as f64 * 0.1).collect();
        let bv: Vec<f64> = (1..=n).map(|k| k as f64 * 0.5).collect();
        let a = s.tensor_variable(av, vec![n], false).unwrap();
        let b = s.tensor_variable(bv, vec![n], false).unwrap();
        let ra = s.tensor_mul(re, a).unwrap();
        let ib = s.tensor_mul(im, b).unwrap();
        let sum = s.tensor_add(ra, ib).unwrap();
        let out = s.tensor_sum(sum).unwrap();
        let rep = s.tensor_backward(out).unwrap();
        assert_close(label, s.tensor_gradient(&rep, x).unwrap(), &golden);
    }
}

/// Chains G/H: complex -> ifft(n) -> weighted real/imag loss.
/// Chain G (n=4, exact) is the case whose probe golden comment was stale.
#[test]
fn ifft_backward_matches_torch() {
    for (label, n, gr, gi) in [
        (
            "G ifft n=4",
            4usize,
            vec![0.25, -0.3, -0.05, 0.2],
            vec![1.25, -0.2, -0.25, -0.3],
        ),
        (
            "H ifft pad n=6",
            6usize,
            vec![0.35, -0.483012702, -0.194337567, -0.05],
            vec![1.75, -0.16339746, -0.221132487, -0.25],
        ),
    ] {
        let mut s = s();
        let xr = s
            .tensor_variable(vec![1.0, 2.0, 3.0, 4.0], vec![4], true)
            .unwrap();
        let xi = s
            .tensor_variable(vec![0.5, 1.0, 1.5, 2.0], vec![4], true)
            .unwrap();
        let z = s.tensor_complex(xr, xi).unwrap();
        let y = s.tensor_ifft(z, Some(n)).unwrap();
        let re = s.tensor_real(y).unwrap();
        let im = s.tensor_imag(y).unwrap();
        let av: Vec<f64> = (1..=n).map(|k| k as f64 * 0.1).collect();
        let bv: Vec<f64> = (1..=n).map(|k| k as f64 * 0.5).collect();
        let a = s.tensor_variable(av, vec![n], false).unwrap();
        let b = s.tensor_variable(bv, vec![n], false).unwrap();
        let ra = s.tensor_mul(re, a).unwrap();
        let ib = s.tensor_mul(im, b).unwrap();
        let sum = s.tensor_add(ra, ib).unwrap();
        let out = s.tensor_sum(sum).unwrap();
        let rep = s.tensor_backward(out).unwrap();
        assert_close(
            &format!("{label} xr"),
            s.tensor_gradient(&rep, xr).unwrap(),
            &gr,
        );
        assert_close(
            &format!("{label} xi"),
            s.tensor_gradient(&rep, xi).unwrap(),
            &gi,
        );
    }
}

/// Chain C: view_as_complex -> view_as_real round-trip is differentiable.
#[test]
fn view_as_complex_real_roundtrip_backward_matches_torch() {
    let mut s = s();
    let rr = s
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], true)
        .unwrap();
    let z = s.tensor_view_as_complex(rr).unwrap();
    let r2 = s.tensor_view_as_real(z).unwrap();
    let w = s
        .tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2], false)
        .unwrap();
    let prod = s.tensor_mul(r2, w).unwrap();
    let out = s.tensor_sum(prod).unwrap();
    let rep = s.tensor_backward(out).unwrap();
    assert_close(
        "C",
        s.tensor_gradient(&rep, rr).unwrap(),
        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    );
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NdFft {
    Forward,
    ComplexForward,
    Inverse,
    RealForward,
    RealInverse,
}

impl NdFft {
    fn complex_input(self) -> bool {
        matches!(
            self,
            Self::ComplexForward | Self::Inverse | Self::RealInverse
        )
    }

    fn run(
        self,
        session: &mut FrankenTorchSession,
        input: TensorNodeId,
        sizes: Option<&[usize]>,
        dims: Option<&[usize]>,
        norm: &str,
    ) -> Result<TensorNodeId, AutogradError> {
        match self {
            Self::Forward | Self::ComplexForward => {
                session.tensor_fftn_norm(input, sizes, dims, norm)
            }
            Self::Inverse => session.tensor_ifftn_norm(input, sizes, dims, norm),
            Self::RealForward => session.tensor_rfftn_norm(input, sizes, dims, norm),
            Self::RealInverse => session.tensor_irfftn_norm(input, sizes, dims, norm),
        }
    }
}

const ND_FFTS: [NdFft; 5] = [
    NdFft::Forward,
    NdFft::ComplexForward,
    NdFft::Inverse,
    NdFft::RealForward,
    NdFft::RealInverse,
];

fn coordinates(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut result = vec![0; shape.len()];
    for dim in (0..shape.len()).rev() {
        result[dim] = flat % shape[dim];
        flat /= shape[dim];
    }
    result
}

/// Independent O(input_len * output_len) Fourier definition. This deliberately
/// does not call an FFT, resize helper, or separable per-axis transform. For
/// irfftn, each non-boundary half-spectrum bin contributes its conjugate pair;
/// the phase includes EVERY transformed axis, not just the compressed axis.
fn direct_nd_dft(
    kind: NdFft,
    real: &[f64],
    imag: &[f64],
    input_shape: &[usize],
    logical_shape: &[usize],
    dims: &[usize],
    norm: &str,
) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    let inverse = matches!(kind, NdFft::Inverse | NdFft::RealInverse);
    let n = dims
        .iter()
        .map(|&dim| logical_shape[dim])
        .product::<usize>() as f64;
    let scale = match (norm, inverse) {
        ("ortho", _) => 1.0 / n.sqrt(),
        ("backward", true) | ("forward", false) => 1.0 / n,
        _ => 1.0,
    };
    let last_dim = *dims.last().unwrap();
    let mut output_shape = logical_shape.to_vec();
    if kind == NdFft::RealForward {
        output_shape[last_dim] = logical_shape[last_dim] / 2 + 1;
    }
    let output_len: usize = output_shape.iter().product();
    let mut output_real = vec![0.0; output_len];
    let mut output_imag = vec![0.0; output_len];
    for output in 0..output_len {
        let out_coord = coordinates(output, &output_shape);
        for source in 0..real.len() {
            let in_coord = coordinates(source, input_shape);
            if (0..input_shape.len()).any(|dim| {
                if !dims.contains(&dim) {
                    in_coord[dim] != out_coord[dim]
                } else if kind == NdFft::RealInverse && dim == last_dim {
                    in_coord[dim] > logical_shape[dim] / 2
                } else {
                    in_coord[dim] >= logical_shape[dim]
                }
            }) {
                continue;
            }
            let phase = std::f64::consts::TAU
                * dims
                    .iter()
                    .map(|&dim| (in_coord[dim] * out_coord[dim]) as f64 / logical_shape[dim] as f64)
                    .sum::<f64>()
                * if inverse { 1.0 } else { -1.0 };
            let (sin, cos) = phase.sin_cos();
            let k = in_coord[last_dim];
            let final_len = logical_shape[last_dim];
            let paired = kind == NdFft::RealInverse
                && k != 0
                && !(final_len.is_multiple_of(2) && k == final_len / 2);
            let factor = scale * if paired { 2.0 } else { 1.0 };
            output_real[output] += factor * (real[source] * cos - imag[source] * sin);
            if kind != NdFft::RealInverse {
                output_imag[output] += factor * (real[source] * sin + imag[source] * cos);
            }
        }
    }
    (output_shape, output_real, output_imag)
}

fn fft_input(
    session: &mut FrankenTorchSession,
    kind: NdFft,
    shape: &[usize],
    real: &[f64],
    imag: &[f64],
    requires_grad: bool,
) -> (TensorNodeId, TensorNodeId, Option<TensorNodeId>) {
    let real_node = session
        .tensor_variable(real.to_vec(), shape.to_vec(), requires_grad)
        .unwrap();
    if kind.complex_input() {
        let imag_node = session
            .tensor_variable(imag.to_vec(), shape.to_vec(), requires_grad)
            .unwrap();
        (
            session.tensor_complex(real_node, imag_node).unwrap(),
            real_node,
            Some(imag_node),
        )
    } else {
        (real_node, real_node, None)
    }
}

#[test]
fn nd_fft_explicit_sizes_and_default_axes_match_direct_dft() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for kind in ND_FFTS {
            for norm in ["backward", "forward", "ortho"] {
                for requires_grad in [false, true] {
                    for explicit_dims in [false, true] {
                        let shape = [3, 2, 4];
                        let real: Vec<_> =
                            (0..24).map(|i| ((i * 7 % 19) as f64 - 9.0) / 8.0).collect();
                        let imag: Vec<_> = (0..24)
                            .map(|i| {
                                if kind.complex_input() {
                                    ((i * 3 % 17) as f64 - 8.0) / 4.0
                                } else {
                                    0.0
                                }
                            })
                            .collect();
                        let sizes: &[usize] = if explicit_dims { &[3, 5] } else { &[5] };
                        let dims: &[usize] = if explicit_dims { &[2, 0] } else { &[2] };
                        let logical_shape = if explicit_dims { [5, 2, 3] } else { [3, 2, 5] };
                        let mut session = FrankenTorchSession::new(mode);
                        let (input, _, _) =
                            fft_input(&mut session, kind, &shape, &real, &imag, requires_grad);
                        let output = kind
                            .run(
                                &mut session,
                                input,
                                Some(sizes),
                                explicit_dims.then_some(dims),
                                norm,
                            )
                            .unwrap();
                        let (expected_shape, expected_real, expected_imag) =
                            direct_nd_dft(kind, &real, &imag, &shape, &logical_shape, dims, norm);
                        assert_eq!(session.tensor_shape(output).unwrap(), expected_shape);
                        let label = format!(
                            "{mode:?} {kind:?} {norm} grad={requires_grad} dims={explicit_dims}"
                        );
                        if kind == NdFft::RealInverse {
                            assert_close(
                                &label,
                                &session.tensor_values(output).unwrap(),
                                &expected_real,
                            );
                        } else {
                            let real = session.tensor_real(output).unwrap();
                            let imag = session.tensor_imag(output).unwrap();
                            assert_close(
                                &label,
                                &session.tensor_values(real).unwrap(),
                                &expected_real,
                            );
                            assert_close(
                                &label,
                                &session.tensor_values(imag).unwrap(),
                                &expected_imag,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn nd_fft_resized_backward_matches_independent_dft_adjoint() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for kind in ND_FFTS {
            for norm in ["backward", "forward", "ortho"] {
                for (shape, sizes, dims, logical_shape) in [
                    (vec![3, 2, 4], vec![3, 5], vec![2, 0], vec![5, 2, 3]),
                    (vec![2, 3], vec![3, 8], vec![0, 1], vec![3, 8]),
                ] {
                    let input_len: usize = shape.iter().product();
                    let real: Vec<_> = (0..input_len).map(|i| i as f64 / 8.0 - 1.0).collect();
                    let imag: Vec<_> = (0..input_len)
                        .map(|i| {
                            if kind.complex_input() {
                                (i % 5) as f64 / 4.0
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    let (output_shape, _, _) =
                        direct_nd_dft(kind, &real, &imag, &shape, &logical_shape, &dims, norm);
                    let output_len: usize = output_shape.iter().product();
                    let wr: Vec<_> = (0..output_len)
                        .map(|i| ((i * 5 % 13) as f64 - 6.0) / 5.0)
                        .collect();
                    let wi: Vec<_> = (0..output_len)
                        .map(|i| ((i * 11 % 17) as f64 - 8.0) / 4.0)
                        .collect();
                    let mut session = FrankenTorchSession::new(mode);
                    let (input, real_node, imag_node) =
                        fft_input(&mut session, kind, &shape, &real, &imag, true);
                    let output = kind
                        .run(&mut session, input, Some(&sizes), Some(&dims), norm)
                        .unwrap();
                    let out_real = if kind == NdFft::RealInverse {
                        output
                    } else {
                        session.tensor_real(output).unwrap()
                    };
                    let wr_node = session
                        .tensor_variable(wr.clone(), output_shape.clone(), false)
                        .unwrap();
                    let mut weighted = session.tensor_mul(out_real, wr_node).unwrap();
                    if kind != NdFft::RealInverse {
                        let out_imag = session.tensor_imag(output).unwrap();
                        let wi_node = session
                            .tensor_variable(wi.clone(), output_shape, false)
                            .unwrap();
                        let weighted_imag = session.tensor_mul(out_imag, wi_node).unwrap();
                        weighted = session.tensor_add(weighted, weighted_imag).unwrap();
                    }
                    let loss = session.tensor_sum(weighted).unwrap();
                    let report = session.tensor_backward(loss).unwrap();
                    let mut expected_real = Vec::with_capacity(input_len);
                    let mut expected_imag = Vec::with_capacity(input_len);
                    for position in 0..input_len {
                        let mut basis = vec![0.0; input_len];
                        basis[position] = 1.0;
                        let zeros = vec![0.0; input_len];
                        for (re, im, target) in [
                            (&basis, &zeros, &mut expected_real),
                            (&zeros, &basis, &mut expected_imag),
                        ] {
                            let (_, yr, yi) =
                                direct_nd_dft(kind, re, im, &shape, &logical_shape, &dims, norm);
                            target.push(
                                yr.iter().zip(&wr).map(|(y, w)| y * w).sum::<f64>()
                                    + yi.iter().zip(&wi).map(|(y, w)| y * w).sum::<f64>(),
                            );
                        }
                    }
                    let label = format!("{mode:?} {kind:?} {norm} sizes={sizes:?}");
                    assert_close(
                        &label,
                        session.tensor_gradient(&report, real_node).unwrap(),
                        &expected_real,
                    );
                    if let Some(imag_node) = imag_node {
                        assert_close(
                            &label,
                            session.tensor_gradient(&report, imag_node).unwrap(),
                            &expected_imag,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn irfftn_restores_nonseparable_spectra_and_odd_lengths() {
    // An axis of length three exposes the missing cross-axis conjugation that
    // length-two fixtures cannot see (each frequency there is self-mirroring).
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for columns in [4, 5] {
            for norm in ["backward", "forward", "ortho"] {
                for explicit_size in [false, true] {
                    if columns == 5 && !explicit_size {
                        continue;
                    }
                    let shape = [3, columns];
                    let values: Vec<_> = (0..3 * columns)
                        .map(|i| ((i * 7 % 19) as f64 - 9.0) / 8.0)
                        .collect();
                    let mut session = FrankenTorchSession::new(mode);
                    let input = session
                        .tensor_variable(values.clone(), shape.to_vec(), true)
                        .unwrap();
                    let spectrum = session.tensor_rfftn_norm(input, None, None, norm).unwrap();
                    let restored = session
                        .tensor_irfftn_norm(
                            spectrum,
                            explicit_size.then_some(shape.as_slice()),
                            None,
                            norm,
                        )
                        .unwrap();
                    assert_close(
                        "real N-D round trip",
                        &session.tensor_values(restored).unwrap(),
                        &values,
                    );
                    let squared = session.tensor_mul(restored, restored).unwrap();
                    let loss = session.tensor_sum(squared).unwrap();
                    let report = session.tensor_backward(loss).unwrap();
                    assert_close(
                        "real N-D round trip gradient",
                        session.tensor_gradient(&report, input).unwrap(),
                        &values.iter().map(|x| 2.0 * x).collect::<Vec<_>>(),
                    );
                }
            }
        }
    }
}

#[test]
fn nd_fft_validates_sizes_and_dimensions_before_execution() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for kind in ND_FFTS {
            let mut session = FrankenTorchSession::new(mode);
            let (input, _, _) = fft_input(&mut session, kind, &[2, 3], &[0.0; 6], &[0.0; 6], true);
            for (sizes, dims) in [
                (Some(vec![2]), Some(vec![0, 1])),
                (Some(vec![2, 3]), Some(vec![1])),
                (Some(vec![2, 3, 4]), None),
                (None, Some(vec![0, 0])),
                (None, Some(vec![2])),
                (Some(vec![0]), Some(vec![1])),
                (Some(vec![usize::MAX, 2]), Some(vec![0, 1])),
                (Some(vec![usize::MAX / 2]), Some(vec![1])),
            ] {
                assert!(
                    kind.run(
                        &mut session,
                        input,
                        sizes.as_deref(),
                        dims.as_deref(),
                        "backward"
                    )
                    .is_err(),
                    "{kind:?}: sizes={sizes:?}, dims={dims:?}"
                );
            }
            assert!(
                kind.run(&mut session, input, None, None, "invalid")
                    .is_err()
            );
            if matches!(kind, NdFft::RealForward | NdFft::RealInverse) {
                assert!(
                    kind.run(&mut session, input, None, Some(&[]), "backward")
                        .is_err()
                );
            }
        }
    }
}

#[test]
fn nd_fft_resizing_preserves_empty_batches() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for kind in ND_FFTS {
            let mut session = FrankenTorchSession::new(mode);
            let (input, real_node, imag_node) =
                fft_input(&mut session, kind, &[0, 3], &[], &[], true);
            let output = kind
                .run(&mut session, input, Some(&[5]), None, "backward")
                .unwrap();
            let columns = if kind == NdFft::RealForward { 3 } else { 5 };
            assert_eq!(session.tensor_shape(output).unwrap(), vec![0, columns]);
            let real = if kind == NdFft::RealInverse {
                output
            } else {
                session.tensor_real(output).unwrap()
            };
            let loss = session.tensor_sum(real).unwrap();
            let report = session.tensor_backward(loss).unwrap();
            assert_eq!(session.tensor_gradient(&report, real_node).unwrap(), &[]);
            if let Some(imag_node) = imag_node {
                assert_eq!(session.tensor_gradient(&report, imag_node).unwrap(), &[]);
            }
        }
    }
}

#[test]
fn nd_fft_resizing_keeps_f32_and_complex64_inputs_trainable() {
    // The existing FFT kernels compute Complex128 (and irfftn computes F64),
    // including for single-precision input. Resizing must retain that behavior
    // while routing derivatives back to the original F32 parameter tensors.
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for kind in ND_FFTS {
            let mut session = FrankenTorchSession::new(mode);
            let real = vec![0.25, -0.5, 0.75, 1.0, -1.25, 0.5];
            let imag = if kind.complex_input() {
                vec![0.5, 0.25, -0.75, 1.0, 0.5, -0.25]
            } else {
                vec![0.0; 6]
            };
            let real_node = session
                .tensor_variable_f32(real.iter().map(|&x| x as f32).collect(), vec![3, 2], true)
                .unwrap();
            let imag_node = kind.complex_input().then(|| {
                session
                    .tensor_variable_f32(imag.iter().map(|&x| x as f32).collect(), vec![3, 2], true)
                    .unwrap()
            });
            let input = if let Some(imag_node) = imag_node {
                session.tensor_complex(real_node, imag_node).unwrap()
            } else {
                real_node
            };
            let output = kind
                .run(&mut session, input, Some(&[2, 5]), None, "ortho")
                .unwrap();
            assert_eq!(
                session.tensor_dtype(input).unwrap(),
                if kind.complex_input() {
                    DType::Complex64
                } else {
                    DType::F32
                }
            );
            assert_eq!(
                session.tensor_dtype(output).unwrap(),
                if kind == NdFft::RealInverse {
                    DType::F64
                } else {
                    DType::Complex128
                }
            );
            let (shape, expected_real, expected_imag) =
                direct_nd_dft(kind, &real, &imag, &[3, 2], &[2, 5], &[0, 1], "ortho");
            let yr = if kind == NdFft::RealInverse {
                output
            } else {
                session.tensor_real(output).unwrap()
            };
            assert_close(
                "F32 FFT real",
                &session.tensor_values(yr).unwrap(),
                &expected_real,
            );
            if kind != NdFft::RealInverse {
                let yi = session.tensor_imag(output).unwrap();
                assert_close(
                    "F32 FFT imaginary",
                    &session.tensor_values(yi).unwrap(),
                    &expected_imag,
                );
            }
            let weights: Vec<_> = (0..expected_real.len())
                .map(|i| (i % 7) as f64 / 4.0 - 0.5)
                .collect();
            let weight_node = session
                .tensor_variable(weights.clone(), shape, false)
                .unwrap();
            let weighted = session.tensor_mul(yr, weight_node).unwrap();
            let loss = session.tensor_sum(weighted).unwrap();
            let report = session.tensor_backward(loss).unwrap();
            for (node, imaginary) in [(Some(real_node), false), (imag_node, true)] {
                let Some(node) = node else {
                    continue;
                };
                let expected: Vec<_> = (0..6)
                    .map(|position| {
                        let mut basis = vec![0.0; 6];
                        basis[position] = 1.0;
                        let zeros = vec![0.0; 6];
                        let (re, im) = if imaginary {
                            (&zeros, &basis)
                        } else {
                            (&basis, &zeros)
                        };
                        let (_, values, _) =
                            direct_nd_dft(kind, re, im, &[3, 2], &[2, 5], &[0, 1], "ortho");
                        values
                            .iter()
                            .zip(&weights)
                            .map(|(value, weight)| value * weight)
                            .sum::<f64>()
                    })
                    .collect();
                assert_close(
                    "F32 FFT gradient",
                    session.tensor_gradient(&report, node).unwrap(),
                    &expected,
                );
            }
        }
        let mut session = FrankenTorchSession::new(mode);
        let input = session
            .tensor_variable_f32(vec![1.0, 2.0], vec![2], true)
            .unwrap();
        let output = session.tensor_fftn(input, None, Some(&[])).unwrap();
        assert_eq!(session.tensor_dtype(output).unwrap(), DType::Complex128);
        let real = session.tensor_real(output).unwrap();
        let loss = session.tensor_sum(real).unwrap();
        let report = session.tensor_backward(loss).unwrap();
        assert_close(
            "empty transform dimensions retain gradient",
            session.tensor_gradient(&report, input).unwrap(),
            &[1.0, 1.0],
        );
    }
}

#[test]
fn nd_fft_higher_order_bridge_backwards_remain_explicitly_unsupported() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for kind in ND_FFTS {
            let mut session = FrankenTorchSession::new(mode);
            let (input, _, _) = fft_input(&mut session, kind, &[2, 3], &[1.0; 6], &[0.0; 6], true);
            let output = kind
                .run(&mut session, input, Some(&[3, 5]), None, "backward")
                .unwrap();
            let real = if kind == NdFft::RealInverse {
                output
            } else {
                session.tensor_real(output).unwrap()
            };
            let loss = session.tensor_sum(real).unwrap();
            let mut options = BackwardOptions::for_mode(mode);
            options.create_graph = true;
            let error = session
                .tensor_backward_with_options(loss, options)
                .unwrap_err();
            assert!(
                error.to_string().contains("create_graph"),
                "{kind:?}: {error}"
            );
        }
    }
}
