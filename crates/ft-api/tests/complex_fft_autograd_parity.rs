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
use ft_core::{
    BFloat16, Complex64, Complex128, DType, DenseTensor, Device, ExecutionMode, Float16,
    TensorMeta, TensorStorage,
};
use std::sync::Arc;

fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_fft_close(label, got, want, 1e-9);
}

fn assert_fft_close(label: &str, got: &[f64], want: &[f64], tolerance: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() <= tolerance,
            "{label}[{i}]: got {g}, want {w}"
        );
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
    HermitianForward,
    HermitianInverse,
}

impl NdFft {
    fn complex_input(self) -> bool {
        matches!(
            self,
            Self::ComplexForward | Self::Inverse | Self::RealInverse | Self::HermitianForward
        )
    }

    fn real_output(self) -> bool {
        matches!(self, Self::RealInverse | Self::HermitianForward)
    }

    fn half_output(self) -> bool {
        matches!(self, Self::RealForward | Self::HermitianInverse)
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
            Self::HermitianForward => session.tensor_hfftn_norm(input, sizes, dims, norm),
            Self::HermitianInverse => session.tensor_ihfftn_norm(input, sizes, dims, norm),
        }
    }
}

const ND_FFTS: [NdFft; 7] = [
    NdFft::Forward,
    NdFft::ComplexForward,
    NdFft::Inverse,
    NdFft::RealForward,
    NdFft::RealInverse,
    NdFft::HermitianForward,
    NdFft::HermitianInverse,
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
/// irfftn/hfftn, each non-boundary half-spectrum bin contributes its conjugate
/// pair; the phase includes EVERY transformed axis, not just the compressed axis.
fn direct_nd_dft(
    kind: NdFft,
    real: &[f64],
    imag: &[f64],
    input_shape: &[usize],
    logical_shape: &[usize],
    dims: &[usize],
    norm: &str,
) -> (Vec<usize>, Vec<f64>, Vec<f64>) {
    let inverse = matches!(
        kind,
        NdFft::Inverse | NdFft::RealInverse | NdFft::HermitianInverse
    );
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
    if kind.half_output() {
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
                } else if kind.real_output() && dim == last_dim {
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
            let paired = kind.real_output()
                && k != 0
                && !(final_len.is_multiple_of(2) && k == final_len / 2);
            let factor = scale * if paired { 2.0 } else { 1.0 };
            output_real[output] += factor * (real[source] * cos - imag[source] * sin);
            if !kind.real_output() {
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
                        if kind.real_output() {
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
                    (vec![2, 3, 4], vec![3, 5, 2], vec![2, 0, 1], vec![5, 2, 3]),
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
                    let out_real = if kind.real_output() {
                        output
                    } else {
                        session.tensor_real(output).unwrap()
                    };
                    let wr_node = session
                        .tensor_variable(wr.clone(), output_shape.clone(), false)
                        .unwrap();
                    let mut weighted = session.tensor_mul(out_real, wr_node).unwrap();
                    if !kind.real_output() {
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
            if kind.half_output() || kind.real_output() {
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
            let columns = if kind.half_output() { 3 } else { 5 };
            assert_eq!(session.tensor_shape(output).unwrap(), vec![0, columns]);
            let real = if kind.real_output() {
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
    // The Hermitian APIs additionally restore the original single precision.
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
                match kind {
                    NdFft::HermitianForward => DType::F32,
                    NdFft::HermitianInverse => DType::Complex64,
                    NdFft::RealInverse => DType::F64,
                    _ => DType::Complex128,
                }
            );
            let (shape, expected_real, expected_imag) =
                direct_nd_dft(kind, &real, &imag, &[3, 2], &[2, 5], &[0, 1], "ortho");
            let yr = if kind.real_output() {
                output
            } else {
                session.tensor_real(output).unwrap()
            };
            let tolerance = if matches!(kind, NdFft::HermitianForward | NdFft::HermitianInverse) {
                2e-6
            } else {
                1e-9
            };
            assert_fft_close(
                "F32 FFT real",
                &session.tensor_values_lossy_f64(yr).unwrap(),
                &expected_real,
                tolerance,
            );
            if !kind.real_output() {
                let yi = session.tensor_imag(output).unwrap();
                assert_fft_close(
                    "F32 FFT imaginary",
                    &session.tensor_values_lossy_f64(yi).unwrap(),
                    &expected_imag,
                    tolerance,
                );
            }
            let weights: Vec<_> = (0..expected_real.len())
                .map(|i| (i % 7) as f64 / 4.0 - 0.5)
                .collect();
            let weight_node = session
                .tensor_variable(weights.clone(), shape, false)
                .unwrap();
            let weight_dtype = session.tensor_dtype(yr).unwrap();
            let weight_node = session.tensor_to_dtype(weight_node, weight_dtype).unwrap();
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
                assert_fft_close(
                    "F32 FFT gradient",
                    session.tensor_gradient(&report, node).unwrap(),
                    &expected,
                    tolerance,
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
            let real = if kind.real_output() {
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

#[test]
fn hermitian_fft_wrappers_preserve_batches_precision_and_training() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for norm in ["backward", "forward", "ortho"] {
                for axes in [1, 2, 3] {
                    let shape = [2, 3, 5];
                    let dims: Vec<_> = (3 - axes..3).collect();
                    let values: Vec<_> =
                        (0..30).map(|i| ((i * 5 % 17) as f64 - 8.0) / 4.0).collect();
                    let mut session = FrankenTorchSession::new(mode);
                    let x = session
                        .tensor_variable(values.clone(), shape.to_vec(), true)
                        .unwrap();
                    let x = session.tensor_to_dtype(x, dtype).unwrap();
                    let spectrum = match (axes, norm) {
                        (1, "backward") => session.tensor_ihfft(x),
                        (2, "backward") => session.tensor_ihfft2(x),
                        (3, "backward") => session.tensor_ihfftn(x, None),
                        (1, _) => session.tensor_ihfft_norm(x, norm),
                        (2, _) => session.tensor_ihfft2_norm(x, None, norm),
                        _ => session.tensor_ihfftn_norm(x, None, None, norm),
                    }
                    .unwrap();
                    assert_eq!(
                        session.tensor_dtype(spectrum).unwrap(),
                        if dtype == DType::F32 {
                            DType::Complex64
                        } else {
                            DType::Complex128
                        }
                    );
                    let (expected_shape, expected_real, expected_imag) = direct_nd_dft(
                        NdFft::HermitianInverse,
                        &values,
                        &[0.0; 30],
                        &shape,
                        &shape,
                        &dims,
                        norm,
                    );
                    assert_eq!(session.tensor_shape(spectrum).unwrap(), expected_shape);
                    let tolerance = if dtype == DType::F32 { 4e-6 } else { 1e-9 };
                    let real = session.tensor_real(spectrum).unwrap();
                    let imag = session.tensor_imag(spectrum).unwrap();
                    assert_fft_close(
                        "Hermitian wrapper real",
                        &session.tensor_values_lossy_f64(real).unwrap(),
                        &expected_real,
                        tolerance,
                    );
                    assert_fft_close(
                        "Hermitian wrapper imaginary",
                        &session.tensor_values_lossy_f64(imag).unwrap(),
                        &expected_imag,
                        tolerance,
                    );
                    let restored = match (axes, norm) {
                        (1, "backward") => session.tensor_hfft(spectrum, Some(5)),
                        (2, "backward") => session.tensor_hfft2(spectrum, Some((3, 5))),
                        (3, "backward") => {
                            session.tensor_hfftn(spectrum, Some(shape.to_vec()), None)
                        }
                        (1, _) => session.tensor_hfft_norm(spectrum, Some(5), norm),
                        (2, _) => session.tensor_hfft2_norm(spectrum, Some((3, 5)), norm),
                        _ => session.tensor_hfftn_norm(spectrum, Some(&shape), None, norm),
                    }
                    .unwrap();
                    assert_eq!(session.tensor_shape(restored).unwrap(), shape);
                    assert_eq!(session.tensor_dtype(restored).unwrap(), dtype);
                    assert_fft_close(
                        "Hermitian wrapper round trip",
                        &session.tensor_values_lossy_f64(restored).unwrap(),
                        &values,
                        tolerance,
                    );
                    let squared = session.tensor_mul(restored, restored).unwrap();
                    let loss = session.tensor_sum(squared).unwrap();
                    let report = session.tensor_backward(loss).unwrap();
                    assert_fft_close(
                        "Hermitian wrapper round trip gradient",
                        session.tensor_gradient(&report, x).unwrap(),
                        &values.iter().map(|value| 2.0 * value).collect::<Vec<_>>(),
                        tolerance,
                    );
                    let inferred_even = session.tensor_hfftn(spectrum, None, Some(dims)).unwrap();
                    assert_eq!(session.tensor_shape(inferred_even).unwrap(), vec![2, 3, 4]);
                }
            }
        }
    }
}

#[test]
fn hermitian_fft_real_input_and_no_grad_match_direct_dft() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for norm in ["backward", "forward", "ortho"] {
                let shape = [2, 3];
                let values = [0.5, -1.0, 0.25, 2.0, 0.75, -0.5];
                let mut session = FrankenTorchSession::new(mode);
                let x = session
                    .tensor_variable(values.to_vec(), shape.to_vec(), true)
                    .unwrap();
                let x = session.tensor_to_dtype(x, dtype).unwrap();
                let output = session
                    .with_no_grad(|session| session.tensor_hfft_norm(x, Some(5), norm))
                    .unwrap();
                assert!(!session.tensor_requires_grad(output).unwrap());
                assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
                let (expected_shape, expected_real, _) = direct_nd_dft(
                    NdFft::HermitianForward,
                    &values,
                    &[0.0; 6],
                    &shape,
                    &[2, 5],
                    &[1],
                    norm,
                );
                assert_eq!(session.tensor_shape(output).unwrap(), expected_shape);
                assert_fft_close(
                    "real Hermitian input",
                    &session.tensor_values_lossy_f64(output).unwrap(),
                    &expected_real,
                    if dtype == DType::F32 { 1e-6 } else { 1e-9 },
                );
                let inverse = session
                    .with_no_grad(|session| session.tensor_ihfftn_norm(x, Some(&[4]), None, norm))
                    .unwrap();
                assert!(!session.tensor_requires_grad(inverse).unwrap());
                assert_eq!(session.tensor_shape(inverse).unwrap(), vec![2, 3]);
                assert_eq!(
                    session.tensor_dtype(inverse).unwrap(),
                    if dtype == DType::F32 {
                        DType::Complex64
                    } else {
                        DType::Complex128
                    }
                );
            }
        }
    }
}

#[test]
fn hermitian_fft_preflights_shapes_dtypes_and_scalar_inputs() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        let mut session = FrankenTorchSession::new(mode);
        let scalar = session.tensor_variable(vec![1.0], vec![], true).unwrap();
        let empty = session.tensor_variable(vec![], vec![0], true).unwrap();
        let singleton = session.tensor_variable(vec![2.0], vec![1], true).unwrap();
        let half = session.tensor_to_dtype(singleton, DType::F16).unwrap();
        let complex = session.tensor_cdouble(singleton).unwrap();
        let initial_nodes = session.autograd_graph_node_count();
        assert!(session.tensor_hfft(scalar, None).is_err());
        assert!(session.tensor_ihfft(scalar).is_err());
        assert!(session.tensor_hfft2(singleton, None).is_err());
        assert!(session.tensor_ihfft2(singleton).is_err());
        assert!(session.tensor_hfft(empty, None).is_err());
        assert!(session.tensor_ihfft(empty).is_err());
        assert!(session.tensor_hfft(singleton, None).is_err());
        assert!(session.tensor_hfft(singleton, Some(0)).is_err());
        assert!(session.tensor_hfft(half, Some(1)).is_err());
        assert!(session.tensor_ihfft(half).is_err());
        assert!(session.tensor_ihfft(complex).is_err());
        assert!(
            session
                .tensor_hfft_norm(complex, Some(1), "invalid")
                .is_err()
        );
        assert!(session.tensor_ihfft_norm(singleton, "invalid").is_err());
        assert_eq!(session.autograd_graph_node_count(), initial_nodes);
        let forward = session.tensor_hfft(singleton, Some(1)).unwrap();
        assert_close(
            "length-one Hermitian forward",
            &session.tensor_values(forward).unwrap(),
            &[2.0],
        );
        let inverse = session.tensor_ihfft(singleton).unwrap();
        let real = session.tensor_real(inverse).unwrap();
        let imag = session.tensor_imag(inverse).unwrap();
        assert_close(
            "length-one Hermitian inverse real",
            &session.tensor_values(real).unwrap(),
            &[2.0],
        );
        assert_close(
            "length-one Hermitian inverse imaginary",
            &session.tensor_values(imag).unwrap(),
            &[0.0],
        );
    }
}

#[test]
fn irfft2_sizes_precision_and_gradients_match_direct_dft() {
    // Three rows expose cross-axis conjugation: unlike length two, frequency
    // row one must pair with a DIFFERENT row. Include arbitrary boundary bins,
    // odd output widths, simultaneous crop/pad, and a length-one transform.
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for input_shape in [vec![3, 3], vec![2, 3, 3]] {
                let input_len: usize = input_shape.iter().product();
                let rank = input_shape.len();
                let dims = [rank - 2, rank - 1];
                let real: Vec<_> = (0..input_len)
                    .map(|i| ((i * 7 % 19) as f64 - 9.0) / 8.0)
                    .collect();
                let imag: Vec<_> = (0..input_len)
                    .map(|i| ((i * 3 % 17) as f64 - 8.0) / 4.0)
                    .collect();
                for norm in ["backward", "forward", "ortho"] {
                    for sizes in [None, Some((2, 5)), Some((4, 6)), Some((1, 1))] {
                        let (rows, columns) = sizes.unwrap_or((3, 4));
                        let mut logical_shape = input_shape.clone();
                        logical_shape[rank - 2] = rows;
                        logical_shape[rank - 1] = columns;
                        let (output_shape, expected, _) = direct_nd_dft(
                            NdFft::RealInverse,
                            &real,
                            &imag,
                            &input_shape,
                            &logical_shape,
                            &dims,
                            norm,
                        );
                        // Frozen inputs, trainable inputs, and trainable inputs
                        // inside no_grad must all have the same forward meaning.
                        for execution in 0..3 {
                            let mut session = FrankenTorchSession::new(mode);
                            let requires_grad = execution != 0;
                            let (re, im) = if dtype == DType::F32 {
                                (
                                    session.tensor_variable_f32(
                                        real.iter().map(|&x| x as f32).collect(),
                                        input_shape.clone(),
                                        requires_grad,
                                    ),
                                    session.tensor_variable_f32(
                                        imag.iter().map(|&x| x as f32).collect(),
                                        input_shape.clone(),
                                        requires_grad,
                                    ),
                                )
                            } else {
                                (
                                    session.tensor_variable(
                                        real.clone(),
                                        input_shape.clone(),
                                        requires_grad,
                                    ),
                                    session.tensor_variable(
                                        imag.clone(),
                                        input_shape.clone(),
                                        requires_grad,
                                    ),
                                )
                            };
                            let (re, im) = (re.unwrap(), im.unwrap());
                            let input = session.tensor_complex(re, im).unwrap();
                            let output = if execution == 2 {
                                session.with_no_grad(|session| {
                                    session.tensor_irfft2_norm(input, sizes, norm)
                                })
                            } else if norm == "backward" {
                                session.tensor_irfft2(input, sizes)
                            } else {
                                session.tensor_irfft2_norm(input, sizes, norm)
                            }
                            .unwrap();
                            let label = format!(
                                "{mode:?} {dtype:?} {input_shape:?} {sizes:?} {norm} execution={execution}"
                            );
                            let tolerance = if dtype == DType::F32 { 6e-6 } else { 1e-9 };
                            assert_eq!(
                                session.tensor_shape(output).unwrap(),
                                output_shape,
                                "{label}"
                            );
                            assert_eq!(session.tensor_dtype(output).unwrap(), dtype, "{label}");
                            assert_eq!(
                                session.tensor_requires_grad(output).unwrap(),
                                execution == 1,
                                "{label}"
                            );
                            assert_fft_close(
                                &label,
                                &session.tensor_values_lossy_f64(output).unwrap(),
                                &expected,
                                tolerance,
                            );
                            if execution != 1 {
                                continue;
                            }
                            let weights: Vec<_> = (0..expected.len())
                                .map(|i| (i % 7) as f64 / 4.0 - 0.5)
                                .collect();
                            let weight = session
                                .tensor_variable(weights.clone(), output_shape.clone(), false)
                                .unwrap();
                            let weight = session.tensor_to_dtype(weight, dtype).unwrap();
                            let weighted = session.tensor_mul(output, weight).unwrap();
                            let loss = session.tensor_sum(weighted).unwrap();
                            let report = session.tensor_backward(loss).unwrap();
                            for (parameter, imaginary) in [(re, false), (im, true)] {
                                let expected_gradient: Vec<_> = (0..input_len)
                                    .map(|position| {
                                        let mut basis = vec![0.0; input_len];
                                        basis[position] = 1.0;
                                        let zeros = vec![0.0; input_len];
                                        let (real_basis, imag_basis) = if imaginary {
                                            (&zeros, &basis)
                                        } else {
                                            (&basis, &zeros)
                                        };
                                        let (_, values, _) = direct_nd_dft(
                                            NdFft::RealInverse,
                                            real_basis,
                                            imag_basis,
                                            &input_shape,
                                            &logical_shape,
                                            &dims,
                                            norm,
                                        );
                                        values.iter().zip(&weights).map(|(y, w)| y * w).sum::<f64>()
                                    })
                                    .collect();
                                assert_fft_close(
                                    &label,
                                    session.tensor_gradient(&report, parameter).unwrap(),
                                    &expected_gradient,
                                    tolerance,
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
fn irfft2_real_input_promotion_preserves_precision_and_gradients() {
    // PyTorch promotes real spectra to complex before the inverse transform.
    // Check that this promotion is differentiable and does not promote F32
    // models to F64 output, including when gradient recording is disabled.
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for norm in ["backward", "forward", "ortho"] {
                let mut session = FrankenTorchSession::new(mode);
                let values: Vec<_> = (0..9).map(|i| i as f64 / 4.0 - 1.0).collect();
                let input = session
                    .tensor_variable(values.clone(), vec![3, 3], true)
                    .unwrap();
                let input = session.tensor_to_dtype(input, dtype).unwrap();
                let output = session
                    .tensor_irfft2_norm(input, Some((3, 5)), norm)
                    .unwrap();
                let (_, expected, _) = direct_nd_dft(
                    NdFft::RealInverse,
                    &values,
                    &[0.0; 9],
                    &[3, 3],
                    &[3, 5],
                    &[0, 1],
                    norm,
                );
                let tolerance = if dtype == DType::F32 { 6e-6 } else { 1e-9 };
                assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
                assert_fft_close(
                    "real irfft2 promotion",
                    &session.tensor_values_lossy_f64(output).unwrap(),
                    &expected,
                    tolerance,
                );
                let weights: Vec<_> = (0..15).map(|i| (i % 5) as f64 / 4.0 - 0.5).collect();
                let weight = session
                    .tensor_variable(weights.clone(), vec![3, 5], false)
                    .unwrap();
                let weight = session.tensor_to_dtype(weight, dtype).unwrap();
                let weighted = session.tensor_mul(output, weight).unwrap();
                let loss = session.tensor_sum(weighted).unwrap();
                let report = session.tensor_backward(loss).unwrap();
                let expected_gradient: Vec<_> = (0..9)
                    .map(|i| {
                        let mut basis = vec![0.0; 9];
                        basis[i] = 1.0;
                        let (_, y, _) = direct_nd_dft(
                            NdFft::RealInverse,
                            &basis,
                            &[0.0; 9],
                            &[3, 3],
                            &[3, 5],
                            &[0, 1],
                            norm,
                        );
                        y.iter().zip(&weights).map(|(y, w)| y * w).sum::<f64>()
                    })
                    .collect();
                assert_fft_close(
                    "real irfft2 adjoint",
                    session.tensor_gradient(&report, input).unwrap(),
                    &expected_gradient,
                    tolerance,
                );
                let frozen = session
                    .with_no_grad(|session| session.tensor_irfft2_norm(input, Some((3, 5)), norm))
                    .unwrap();
                assert!(!session.tensor_requires_grad(frozen).unwrap());
                assert_eq!(session.tensor_dtype(frozen).unwrap(), dtype);
                assert_fft_close(
                    "real irfft2 no_grad",
                    &session.tensor_values_lossy_f64(frozen).unwrap(),
                    &expected,
                    tolerance,
                );
            }
        }
    }
}

#[test]
fn irfft2_preserves_empty_batches_and_preflights_invalid_arguments() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            let mut session = FrankenTorchSession::new(mode);
            let empty = session
                .tensor_variable(vec![], vec![0, 3, 3], true)
                .unwrap();
            let empty = session.tensor_to_dtype(empty, dtype).unwrap();
            let spectrum = session.tensor_complex(empty, empty).unwrap();
            let output = session
                .tensor_irfft2_norm(spectrum, Some((4, 5)), "ortho")
                .unwrap();
            assert_eq!(session.tensor_shape(output).unwrap(), vec![0, 4, 5]);
            assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
            let loss = session.tensor_sum(output).unwrap();
            let report = session.tensor_backward(loss).unwrap();
            assert!(session.tensor_gradient(&report, empty).unwrap().is_empty());
        }

        let mut session = FrankenTorchSession::new(mode);
        let real = session
            .tensor_variable(vec![1.0; 9], vec![3, 3], true)
            .unwrap();
        let spectrum = session.tensor_complex(real, real).unwrap();
        let scalar = session.tensor_variable(vec![1.0], vec![], true).unwrap();
        let singleton = session.tensor_variable(vec![1.0], vec![1], true).unwrap();
        let scalar = session.tensor_complex(scalar, scalar).unwrap();
        let singleton = session.tensor_complex(singleton, singleton).unwrap();
        let initial_nodes = session.autograd_graph_node_count();
        for (sizes, norm) in [
            (Some((0, 4)), "backward"),
            (Some((3, 0)), "backward"),
            (Some((usize::MAX, 4)), "backward"),
            (None, "invalid"),
        ] {
            assert!(session.tensor_irfft2_norm(spectrum, sizes, norm).is_err());
            assert!(session.tensor_irfft2_norm(real, sizes, norm).is_err());
        }
        for invalid in [scalar, singleton] {
            assert!(session.tensor_irfft2(invalid, None).is_err());
        }
        assert_eq!(session.autograd_graph_node_count(), initial_nodes);
    }
}

#[test]
fn irfft2_offset_and_strided_views_preserve_values_and_gradients() {
    use ft_core::{Complex64, Complex128, DenseTensor, Device, TensorMeta};

    let real: Vec<_> = (0..9).map(|i| i as f64 / 4.0 - 1.0).collect();
    let imag: Vec<_> = (0..9).map(|i| (i % 5) as f64 / 8.0 - 0.25).collect();
    let weights: Vec<_> = (0..15).map(|i| (i % 7) as f64 / 4.0 - 0.5).collect();
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for complex in [false, true] {
                let input_imag = if complex { imag.clone() } else { vec![0.0; 9] };
                let (_, expected, _) = direct_nd_dft(
                    NdFft::RealInverse,
                    &real,
                    &input_imag,
                    &[3, 3],
                    &[3, 5],
                    &[0, 1],
                    "ortho",
                );
                let mut expected_gradient = Vec::new();
                for position in 0..9 {
                    for component in 0..if complex { 2 } else { 1 } {
                        let mut basis = vec![0.0; 9];
                        basis[position] = 1.0;
                        let zeros = vec![0.0; 9];
                        let (re, im) = if component == 0 {
                            (&basis, &zeros)
                        } else {
                            (&zeros, &basis)
                        };
                        let (_, output, _) = direct_nd_dft(
                            NdFft::RealInverse,
                            re,
                            im,
                            &[3, 3],
                            &[3, 5],
                            &[0, 1],
                            "ortho",
                        );
                        expected_gradient
                            .push(output.iter().zip(&weights).map(|(y, w)| y * w).sum::<f64>());
                    }
                }
                for strided in [false, true] {
                    let (strides, offset) = if strided {
                        (vec![1, 4], 2)
                    } else {
                        (vec![3, 1], 9)
                    };
                    let source_dtype = match (dtype, complex) {
                        (DType::F32, true) => DType::Complex64,
                        (DType::F64, true) => DType::Complex128,
                        _ => dtype,
                    };
                    let meta = TensorMeta::from_shape_and_strides(
                        vec![3, 3],
                        strides.clone(),
                        offset,
                        source_dtype,
                        Device::Cpu,
                    )
                    .unwrap();
                    // Values outside the view are deliberately unlike its logical data.
                    let mut physical_real = vec![91.0; 18];
                    let mut physical_imag = vec![-37.0; 18];
                    for row in 0..3 {
                        for col in 0..3 {
                            let physical = offset + row * strides[0] + col * strides[1];
                            physical_real[physical] = real[row * 3 + col];
                            physical_imag[physical] = input_imag[row * 3 + col];
                        }
                    }
                    let source = match source_dtype {
                        DType::F32 => DenseTensor::from_storage_f32(
                            meta.clone(),
                            physical_real.iter().map(|&x| x as f32).collect(),
                        ),
                        DType::F64 => {
                            DenseTensor::from_storage(meta.clone(), physical_real.clone())
                        }
                        DType::Complex64 => DenseTensor::from_storage_complex64(
                            meta.clone(),
                            physical_real
                                .iter()
                                .zip(&physical_imag)
                                .map(|(&re, &im)| Complex64::new(re as f32, im as f32))
                                .collect(),
                        ),
                        DType::Complex128 => DenseTensor::from_storage_complex128(
                            meta.clone(),
                            physical_real
                                .iter()
                                .zip(&physical_imag)
                                .map(|(&re, &im)| Complex128::new(re, im))
                                .collect(),
                        ),
                        _ => unreachable!(),
                    }
                    .unwrap();
                    for execution in 0..3 {
                        let mut session = FrankenTorchSession::new(mode);
                        let input =
                            session.tensor_variable_from_storage(source.clone(), execution != 0);
                        let output = if execution == 2 {
                            session.with_no_grad(|s| {
                                s.tensor_irfft2_norm(input, Some((3, 5)), "ortho")
                            })
                        } else {
                            session.tensor_irfft2_norm(input, Some((3, 5)), "ortho")
                        }
                        .unwrap();
                        let label = format!(
                            "{mode:?} {source_dtype:?}, strided={strided}, execution={execution}"
                        );
                        let tolerance = if dtype == DType::F32 { 6e-6 } else { 1e-9 };
                        assert_eq!(session.tensor_dtype(output).unwrap(), dtype, "{label}");
                        assert_eq!(
                            session.tensor_requires_grad(output).unwrap(),
                            execution == 1,
                            "{label}"
                        );
                        assert!(session.is_grad_enabled());
                        assert_eq!(session.tensor_snapshot(input).unwrap().meta(), &meta);
                        assert_fft_close(
                            &label,
                            &session.tensor_values_lossy_f64(output).unwrap(),
                            &expected,
                            tolerance,
                        );
                        if execution == 1 {
                            let weight = session
                                .tensor_variable(weights.clone(), vec![3, 5], false)
                                .unwrap();
                            let weight = session.tensor_to_dtype(weight, dtype).unwrap();
                            let product = session.tensor_mul(output, weight).unwrap();
                            let loss = session.tensor_sum(product).unwrap();
                            let report = session.tensor_backward(loss).unwrap();
                            assert_fft_close(
                                &label,
                                report.gradient(input).unwrap(),
                                &expected_gradient,
                                tolerance,
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn irfft2_narrow_views_backpropagate_to_full_parameters() {
    let base_real: Vec<_> = (0..18).map(|i| i as f64 / 4.0 - 2.0).collect();
    let base_imag: Vec<_> = (0..18).map(|i| (i % 7) as f64 / 8.0 - 0.5).collect();
    let weights: Vec<_> = (0..12).map(|i| (i % 5) as f64 / 4.0 - 0.25).collect();
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for complex in [false, true] {
                for start in [0, 1] {
                    let mut session = FrankenTorchSession::new(mode);
                    let re = session
                        .tensor_variable(base_real.clone(), vec![2, 3, 3], true)
                        .unwrap();
                    let re = session.tensor_to_dtype(re, dtype).unwrap();
                    let im = session
                        .tensor_variable(base_imag.clone(), vec![2, 3, 3], true)
                        .unwrap();
                    let im = session.tensor_to_dtype(im, dtype).unwrap();
                    let base = if complex {
                        session.tensor_complex(re, im).unwrap()
                    } else {
                        re
                    };
                    let view = session.tensor_narrow(base, 0, start, 1).unwrap();
                    let output = session.tensor_irfft2(view, None).unwrap();
                    let input_imag = if complex {
                        base_imag[start * 9..start * 9 + 9].to_vec()
                    } else {
                        vec![0.0; 9]
                    };
                    let (_, expected, _) = direct_nd_dft(
                        NdFft::RealInverse,
                        &base_real[start * 9..start * 9 + 9],
                        &input_imag,
                        &[1, 3, 3],
                        &[1, 3, 4],
                        &[1, 2],
                        "backward",
                    );
                    let tolerance = if dtype == DType::F32 { 6e-6 } else { 1e-9 };
                    let label =
                        format!("{mode:?} {dtype:?}, complex={complex}, narrow start={start}");
                    assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
                    assert_fft_close(
                        &label,
                        &session.tensor_values_lossy_f64(output).unwrap(),
                        &expected,
                        tolerance,
                    );
                    let weight = session
                        .tensor_variable(weights.clone(), vec![1, 3, 4], false)
                        .unwrap();
                    let weight = session.tensor_to_dtype(weight, dtype).unwrap();
                    let product = session.tensor_mul(output, weight).unwrap();
                    let loss = session.tensor_sum(product).unwrap();
                    let report = session.tensor_backward(loss).unwrap();
                    for (parameter, imaginary) in [(re, false), (im, true)] {
                        if imaginary && !complex {
                            continue;
                        }
                        let mut expected_gradient = vec![0.0; 18];
                        for position in 0..9 {
                            let mut basis = vec![0.0; 9];
                            basis[position] = 1.0;
                            let zeros = vec![0.0; 9];
                            let (real, imag) = if imaginary {
                                (&zeros, &basis)
                            } else {
                                (&basis, &zeros)
                            };
                            let (_, values, _) = direct_nd_dft(
                                NdFft::RealInverse,
                                real,
                                imag,
                                &[1, 3, 3],
                                &[1, 3, 4],
                                &[1, 2],
                                "backward",
                            );
                            expected_gradient[start * 9 + position] =
                                values.iter().zip(&weights).map(|(y, w)| y * w).sum();
                        }
                        assert_fft_close(
                            &label,
                            report.gradient(parameter).unwrap(),
                            &expected_gradient,
                            tolerance,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn complex_narrow_vjp_preserves_both_lanes_on_every_axis() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for dim in 0..3 {
                let mut session = FrankenTorchSession::new(mode);
                let re = session
                    .tensor_variable(vec![1.0; 24], vec![2, 3, 4], true)
                    .unwrap();
                let re = session.tensor_to_dtype(re, dtype).unwrap();
                let im = session
                    .tensor_variable(vec![2.0; 24], vec![2, 3, 4], true)
                    .unwrap();
                let im = session.tensor_to_dtype(im, dtype).unwrap();
                let base = session.tensor_complex(re, im).unwrap();
                let view = session.tensor_narrow(base, dim, 1, 1).unwrap();
                let shape = session.tensor_shape(view).unwrap();
                let numel: usize = shape.iter().product();
                let seed: Vec<_> = (0..2 * numel).map(|i| (i + 1) as f64 / 4.0).collect();
                let gradients = session
                    .tensor_autograd_grad(
                        &[view],
                        &[re, im],
                        Some(std::slice::from_ref(&seed)),
                        false,
                        false,
                    )
                    .unwrap();
                let mut expected_real = vec![0.0; 24];
                let mut expected_imag = vec![0.0; 24];
                for source in 0..24 {
                    let mut coord = coordinates(source, &[2, 3, 4]);
                    if coord[dim] == 1 {
                        coord[dim] = 0;
                        let destination = coord
                            .iter()
                            .zip(&shape)
                            .fold(0, |flat, (&c, &size)| flat * size + c);
                        expected_real[source] = seed[2 * destination];
                        expected_imag[source] = seed[2 * destination + 1];
                    }
                }
                assert_fft_close(
                    "complex narrow real VJP",
                    gradients[0].as_deref().unwrap(),
                    &expected_real,
                    0.0,
                );
                assert_fft_close(
                    "complex narrow imaginary VJP",
                    gradients[1].as_deref().unwrap(),
                    &expected_imag,
                    0.0,
                );
            }
        }
    }
}

fn real_strided_leaf(
    session: &mut FrankenTorchSession,
    dtype: DType,
    values: &[f64],
    shape: &[usize],
    strides: &[usize],
    offset: usize,
    requires_grad: bool,
) -> TensorNodeId {
    let storage = match dtype {
        DType::F64 => TensorStorage::F64(Arc::new(values.to_vec())),
        DType::F32 => {
            TensorStorage::F32(Arc::new(values.iter().map(|&value| value as f32).collect()))
        }
        DType::F16 => TensorStorage::F16(Arc::new(
            values
                .iter()
                .map(|&value| Float16::from_f32(value as f32))
                .collect(),
        )),
        DType::BF16 => TensorStorage::BF16(Arc::new(
            values
                .iter()
                .map(|&value| BFloat16::from_f32(value as f32))
                .collect(),
        )),
        _ => unreachable!(),
    };
    let meta = TensorMeta::from_shape_and_strides(
        shape.to_vec(),
        strides.to_vec(),
        offset,
        dtype,
        Device::Cpu,
    )
    .unwrap();
    session.tensor_variable_from_storage(
        DenseTensor::from_typed_storage(meta, storage).unwrap(),
        requires_grad,
    )
}

#[test]
fn complex_components_read_offset_views_and_route_weighted_gradients() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            for start in [0, 1] {
                for operation in 0..4 {
                    let mut session = FrankenTorchSession::new(mode);
                    let re = real_strided_leaf(
                        &mut session,
                        dtype,
                        &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                        &[4, 2],
                        &[2, 1],
                        0,
                        true,
                    );
                    let im = real_strided_leaf(
                        &mut session,
                        dtype,
                        &[11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0],
                        &[4, 2],
                        &[2, 1],
                        0,
                        true,
                    );
                    let z = session.tensor_complex(re, im).unwrap();
                    let source = session.tensor_snapshot(z).unwrap();
                    let view = session.tensor_narrow(z, 0, start, 2).unwrap();
                    let output = match operation {
                        0 => session.tensor_real(view).unwrap(),
                        1 => session.tensor_imag(view).unwrap(),
                        2 => session.tensor_view_as_real(view).unwrap(),
                        _ => {
                            let conjugate = session.tensor_conj(view).unwrap();
                            session.tensor_view_as_real(conjugate).unwrap()
                        }
                    };
                    let mut expected = Vec::new();
                    for index in start * 2..start * 2 + 4 {
                        let real = index as f64 + 1.0;
                        let imag = index as f64 + 11.0;
                        match operation {
                            0 => expected.push(real),
                            1 => expected.push(imag),
                            2 => expected.extend([real, imag]),
                            _ => expected.extend([real, -imag]),
                        }
                    }
                    assert_close(
                        "logical complex view",
                        &session.tensor_values_lossy_f64(output).unwrap(),
                        &expected,
                    );
                    let weights: Vec<_> = (1..=expected.len()).map(|value| value as f64).collect();
                    let gradients = session
                        .tensor_autograd_grad(
                            &[output],
                            &[re, im],
                            Some(&[weights.clone()]),
                            false,
                            false,
                        )
                        .unwrap();
                    let mut expected_re = vec![0.0; 8];
                    let mut expected_im = vec![0.0; 8];
                    for index in 0..4 {
                        let position = start * 2 + index;
                        match operation {
                            0 => expected_re[position] = weights[index],
                            1 => expected_im[position] = weights[index],
                            _ => {
                                expected_re[position] = weights[2 * index];
                                expected_im[position] = weights[2 * index + 1]
                                    * if operation == 3 { -1.0 } else { 1.0 };
                            }
                        }
                    }
                    assert_close(
                        "offset real VJP",
                        gradients[0].as_ref().unwrap(),
                        &expected_re,
                    );
                    assert_close(
                        "offset imaginary VJP",
                        gradients[1].as_ref().unwrap(),
                        &expected_im,
                    );
                    let after = session.tensor_snapshot(z).unwrap();
                    assert_eq!(after.meta(), source.meta());
                    assert_eq!(
                        scalar_storage_bits(after.typed_storage()),
                        scalar_storage_bits(source.typed_storage())
                    );
                }
            }
        }
    }
}

#[test]
fn complex_assembly_reads_independent_strides_offsets_and_preserves_vjps() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            let mut session = FrankenTorchSession::new(mode);
            let re = real_strided_leaf(
                &mut session,
                dtype,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                &[2, 2],
                &[3, 1],
                1,
                true,
            );
            let im = real_strided_leaf(
                &mut session,
                dtype,
                &[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0],
                &[2, 2],
                &[1, 3],
                2,
                true,
            );
            let z = session.tensor_complex(re, im).unwrap();
            let output = session.tensor_view_as_real(z).unwrap();
            assert_close(
                "independent strided complex components",
                &session.tensor_values_lossy_f64(output).unwrap(),
                &[1.0, 20.0, 2.0, 50.0, 4.0, 30.0, 5.0, 60.0],
            );
            let gradients = session
                .tensor_autograd_grad(
                    &[output],
                    &[re, im],
                    Some(&[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]]),
                    false,
                    false,
                )
                .unwrap();
            assert_close(
                "strided real VJP",
                gradients[0].as_ref().unwrap(),
                &[1.0, 3.0, 5.0, 7.0],
            );
            assert_close(
                "strided imaginary VJP",
                gradients[1].as_ref().unwrap(),
                &[2.0, 4.0, 6.0, 8.0],
            );
        }
    }
}

#[test]
fn complex_view_roundtrip_reads_legal_strides_and_rejects_invalid_layouts_atomically() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            let mut session = FrankenTorchSession::new(mode);
            let values: Vec<_> = (0..16).map(|value| value as f64).collect();
            let input = real_strided_leaf(
                &mut session,
                dtype,
                &values,
                &[2, 2, 2],
                &[8, 2, 1],
                2,
                true,
            );
            let complex = session.tensor_view_as_complex(input).unwrap();
            let output = session.tensor_view_as_real(complex).unwrap();
            assert_close(
                "legal strided view_as_complex",
                &session.tensor_values_lossy_f64(output).unwrap(),
                &[2.0, 3.0, 4.0, 5.0, 10.0, 11.0, 12.0, 13.0],
            );
            let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
            let gradients = session
                .tensor_autograd_grad(&[output], &[input], Some(&[weights.clone()]), false, false)
                .unwrap();
            assert_close(
                "strided view roundtrip VJP",
                gradients[0].as_ref().unwrap(),
                &weights,
            );
            for (strides, offset) in [([2, 1], 1), ([3, 1], 0), ([4, 2], 0)] {
                let input = real_strided_leaf(
                    &mut session,
                    dtype,
                    &values,
                    &[2, 2],
                    &strides,
                    offset,
                    true,
                );
                let before = session.autograd_graph_node_count();
                assert!(session.tensor_view_as_complex(input).is_err());
                assert_eq!(session.autograd_graph_node_count(), before);
            }
        }
    }
}

#[test]
fn contiguous_materializes_strides_with_real_first_and_second_derivatives() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            let mut session = FrankenTorchSession::new(mode);
            let input = real_strided_leaf(
                &mut session,
                dtype,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                &[2, 2],
                &[3, 1],
                1,
                true,
            );
            assert!(!session.tensor_is_contiguous(input).unwrap());
            assert!(!session.is_contiguous(input));
            let output = session.tensor_contiguous(input).unwrap();
            assert_ne!(output, input);
            assert!(session.tensor_is_contiguous(output).unwrap());
            assert_eq!(session.tensor_contiguous(output).unwrap(), output);
            assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
            assert_eq!(
                session
                    .tensor_snapshot(output)
                    .unwrap()
                    .meta()
                    .storage_offset(),
                0
            );
            assert_close(
                "contiguous logical values",
                &session.tensor_values_lossy_f64(output).unwrap(),
                &[1.0, 2.0, 4.0, 5.0],
            );
            let weights = real_strided_leaf(
                &mut session,
                dtype,
                &[0.5, 1.0, 1.5, 2.0],
                &[2, 2],
                &[2, 1],
                0,
                false,
            );
            let square = session.tensor_mul(output, output).unwrap();
            let weighted = session.tensor_mul(square, weights).unwrap();
            let loss = session.tensor_sum(weighted).unwrap();
            let report = session
                .tensor_backward_with_options(
                    loss,
                    BackwardOptions::for_mode(mode).with_create_graph(true),
                )
                .unwrap();
            assert_close(
                "contiguous first derivative",
                report.gradient(input).unwrap(),
                &[1.0, 4.0, 12.0, 20.0],
            );
            let gradient = report.gradient_node(input).unwrap();
            let gradient_sum = session.tensor_sum(gradient).unwrap();
            let second = session.tensor_backward(gradient_sum).unwrap();
            assert_close(
                "contiguous second derivative",
                second.gradient(input).unwrap(),
                &[1.0, 2.0, 3.0, 4.0],
            );

            let detached = session
                .with_no_grad(|session| session.tensor_contiguous(input))
                .unwrap();
            assert!(!session.tensor_requires_grad(detached).unwrap());
            assert!(session.is_grad_enabled());
            let base = session
                .tensor_variable(vec![1.0, 2.0, 3.0], vec![3], true)
                .unwrap();
            let offset = session.tensor_narrow(base, 0, 1, 2).unwrap();
            assert_eq!(
                session
                    .tensor_snapshot(offset)
                    .unwrap()
                    .meta()
                    .storage_offset(),
                1
            );
            assert_eq!(session.tensor_contiguous(offset).unwrap(), offset);
        }
    }
}

fn scalar_storage_bits(storage: &TensorStorage) -> Vec<u64> {
    match storage {
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
        _ => unreachable!(),
    }
}

#[test]
fn contiguous_and_complex_components_preserve_nonfinite_bits_device_and_no_grad() {
    let real32 = vec![
        91.0,
        -0.0,
        91.0,
        f32::from_bits(0x7fc0_1234),
        91.0,
        f32::NEG_INFINITY,
        91.0,
    ];
    let real64 = vec![
        91.0,
        -0.0,
        91.0,
        f64::from_bits(0x7ff8_0000_0000_1234),
        91.0,
        f64::NEG_INFINITY,
        91.0,
    ];
    let storages = [
        TensorStorage::F32(Arc::new(real32.clone())),
        TensorStorage::F64(Arc::new(real64.clone())),
        TensorStorage::F16(Arc::new(vec![
            Float16::from_f32(91.0),
            Float16::from_bits(0x8000),
            Float16::from_f32(91.0),
            Float16::from_bits(0x7e15),
            Float16::from_f32(91.0),
            Float16::from_bits(0xfc00),
            Float16::from_f32(91.0),
        ])),
        TensorStorage::BF16(Arc::new(vec![
            BFloat16::from_f32(91.0),
            BFloat16::from_bits(0x8000),
            BFloat16::from_f32(91.0),
            BFloat16::from_bits(0x7fc5),
            BFloat16::from_f32(91.0),
            BFloat16::from_bits(0xff80),
            BFloat16::from_f32(91.0),
        ])),
        TensorStorage::Complex64(Arc::new(
            real32
                .iter()
                .enumerate()
                .map(|(i, &re)| Complex64::new(re, real32[6 - i]))
                .collect(),
        )),
        TensorStorage::Complex128(Arc::new(
            real64
                .iter()
                .enumerate()
                .map(|(i, &re)| Complex128::new(re, real64[6 - i]))
                .collect(),
        )),
    ];
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for storage in &storages {
            let mut session = FrankenTorchSession::new(mode);
            let dtype = storage.dtype();
            let lanes = if dtype.is_complex() { 2 } else { 1 };
            let source_bits = scalar_storage_bits(storage);
            let expected: Vec<_> = [1, 3, 5]
                .into_iter()
                .flat_map(|index| {
                    source_bits[index * lanes..(index + 1) * lanes]
                        .iter()
                        .copied()
                })
                .collect();
            let meta =
                TensorMeta::from_shape_and_strides(vec![3], vec![2], 1, dtype, Device::Metal)
                    .unwrap();
            let input = session.tensor_variable_from_storage(
                DenseTensor::from_typed_storage(meta, storage.clone()).unwrap(),
                true,
            );
            let output = session
                .with_no_grad(|session| session.tensor_contiguous(input))
                .unwrap();
            let tensor = session.tensor_snapshot(output).unwrap();
            assert_eq!(scalar_storage_bits(tensor.typed_storage()), expected);
            assert_eq!(tensor.meta().dtype(), dtype);
            assert_eq!(tensor.meta().device(), Device::Metal);
            assert!(!session.tensor_requires_grad(output).unwrap());
            if dtype.is_complex() {
                for operation in 0..3 {
                    let output = session
                        .with_no_grad(|session| match operation {
                            0 => session.tensor_real(input),
                            1 => session.tensor_imag(input),
                            _ => session.tensor_view_as_real(input),
                        })
                        .unwrap();
                    let tensor = session.tensor_snapshot(output).unwrap();
                    let expected_component: Vec<_> = if operation == 2 {
                        expected.clone()
                    } else {
                        expected
                            .iter()
                            .skip(operation)
                            .step_by(2)
                            .copied()
                            .collect()
                    };
                    assert_eq!(
                        scalar_storage_bits(tensor.typed_storage()),
                        expected_component
                    );
                    assert_eq!(tensor.meta().device(), Device::Metal);
                    assert!(!session.tensor_requires_grad(output).unwrap());
                }
            }
            assert!(session.is_grad_enabled());
            assert_eq!(
                scalar_storage_bits(session.tensor_snapshot(input).unwrap().typed_storage()),
                source_bits
            );
        }
    }
}

#[test]
fn complex_bridge_preflight_and_higher_order_failures_are_explicit() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        let mut session = FrankenTorchSession::new(mode);
        let real = session.tensor_variable(vec![2.0], vec![1], true).unwrap();
        let misplaced = session.tensor_variable_from_storage(
            DenseTensor::from_contiguous_values(vec![3.0], vec![1], Device::Metal).unwrap(),
            true,
        );
        let before = session.autograd_graph_node_count();
        assert!(session.tensor_complex(real, misplaced).is_err());
        assert_eq!(session.autograd_graph_node_count(), before);
        let invalid = TensorNodeId(usize::MAX);
        assert!(session.tensor_is_contiguous(invalid).is_err());
        assert!(session.tensor_contiguous(invalid).is_err());
        assert!(!session.is_contiguous(invalid));
        assert_eq!(session.tensor_real(real).unwrap(), real);
        let imag = session.tensor_variable(vec![3.0], vec![1], false).unwrap();
        let complex = session.tensor_complex(real, imag).unwrap();
        let view = session.tensor_narrow(complex, 0, 0, 1).unwrap();
        let component = session.tensor_real(view).unwrap();
        let loss = session.tensor_sum(component).unwrap();
        let error = session
            .tensor_backward_with_options(
                loss,
                BackwardOptions::for_mode(mode).with_create_graph(true),
            )
            .unwrap_err();
        assert!(error.to_string().contains("create_graph"));
    }
}

#[test]
fn complex_bridges_handle_scalar_and_empty_logical_views() {
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in [DType::F32, DType::F64] {
            let mut session = FrankenTorchSession::new(mode);
            let re = real_strided_leaf(&mut session, dtype, &[1.0, 2.0, 3.0], &[], &[], 1, true);
            let im = real_strided_leaf(&mut session, dtype, &[1.0, 2.0, 3.0], &[], &[], 2, true);
            let z = session.tensor_complex(re, im).unwrap();
            let output = session.tensor_view_as_real(z).unwrap();
            assert_close(
                "scalar complex components",
                &session.tensor_values_lossy_f64(output).unwrap(),
                &[2.0, 3.0],
            );
            let gradients = session
                .tensor_autograd_grad(&[output], &[re, im], Some(&[vec![4.0, 5.0]]), false, false)
                .unwrap();
            assert_close("scalar real VJP", gradients[0].as_ref().unwrap(), &[4.0]);
            assert_close(
                "scalar imaginary VJP",
                gradients[1].as_ref().unwrap(),
                &[5.0],
            );

            let empty = real_strided_leaf(
                &mut session,
                dtype,
                &[1.0, 2.0, 3.0],
                &[0, 2],
                &[8, 1],
                2,
                true,
            );
            assert!(session.tensor_is_contiguous(empty).unwrap());
            assert_eq!(session.tensor_contiguous(empty).unwrap(), empty);
            let z = session.tensor_view_as_complex(empty).unwrap();
            assert_eq!(session.tensor_shape(z).unwrap(), vec![0]);
            let real = session.tensor_real(z).unwrap();
            let imag = session.tensor_imag(z).unwrap();
            let roundtrip = session.tensor_view_as_real(z).unwrap();
            for output in [real, imag, roundtrip] {
                assert_eq!(session.tensor_dtype(output).unwrap(), dtype);
                assert!(session.tensor_values_lossy_f64(output).unwrap().is_empty());
            }
            assert_eq!(session.tensor_shape(roundtrip).unwrap(), vec![0, 2]);
        }
    }
}

#[test]
fn strided_float_casts_and_custom_functions_keep_logical_values_and_vjps() {
    let dtypes = [DType::F32, DType::F64, DType::F16, DType::BF16];
    for mode in [ExecutionMode::Strict, ExecutionMode::Hardened] {
        for dtype in dtypes {
            for target in dtypes {
                let mut session = FrankenTorchSession::new(mode);
                let input = real_strided_leaf(
                    &mut session,
                    dtype,
                    &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    &[2, 2],
                    &[3, 1],
                    1,
                    true,
                );
                let cast = session.tensor_to_dtype(input, target).unwrap();
                if target == dtype {
                    assert_eq!(cast, input);
                }
                let output = session.tensor_contiguous(cast).unwrap();
                assert_eq!(session.tensor_dtype(output).unwrap(), target);
                assert_close(
                    "strided floating cast",
                    &session.tensor_values_lossy_f64(output).unwrap(),
                    &[1.0, 2.0, 4.0, 5.0],
                );
                let gradients = session
                    .tensor_autograd_grad(
                        &[output],
                        &[input],
                        Some(&[vec![1.0, 2.0, 3.0, 4.0]]),
                        false,
                        false,
                    )
                    .unwrap();
                assert_close(
                    "strided cast VJP",
                    gradients[0].as_ref().unwrap(),
                    &[1.0, 2.0, 3.0, 4.0],
                );
            }
            let mut session = FrankenTorchSession::new(mode);
            let input = real_strided_leaf(
                &mut session,
                dtype,
                &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                &[2, 2],
                &[3, 1],
                1,
                true,
            );
            let output = session
                .tensor_apply_function(
                    &[input],
                    |ctx, inputs| {
                        ctx.save_for_backward(inputs[0].0.to_vec(), inputs[0].1.to_vec());
                        Ok((
                            inputs[0].0.iter().map(|value| value * value).collect(),
                            inputs[0].1.to_vec(),
                        ))
                    },
                    |ctx, gradients| {
                        Ok(vec![Some(
                            ctx.saved_tensors()[0]
                                .iter()
                                .zip(gradients[0])
                                .map(|(value, gradient)| 2.0 * value * gradient)
                                .collect(),
                        )])
                    },
                )
                .unwrap();
            assert_close(
                "strided custom forward",
                &session.tensor_values_lossy_f64(output).unwrap(),
                &[1.0, 4.0, 16.0, 25.0],
            );
            let gradients = session
                .tensor_autograd_grad(
                    &[output],
                    &[input],
                    Some(&[vec![1.0, 2.0, 3.0, 4.0]]),
                    false,
                    false,
                )
                .unwrap();
            assert_close(
                "strided custom VJP",
                gradients[0].as_ref().unwrap(),
                &[2.0, 8.0, 24.0, 40.0],
            );
        }
    }
}
