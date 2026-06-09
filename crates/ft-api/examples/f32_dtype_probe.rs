//! f32 dtype-parity probe (frankentorch project_f32_parity_vein). torch keeps
//! the input float dtype for elementwise + reduction float ops; ft sometimes
//! errors on f32 (tensor_values is f64-only) or silently upcasts to F64. Each
//! op below is run on an f32 input; anything that ERRs or returns dtype != F32
//! is a parity bug.
//!
//!   cargo run -q -p ft-api --example f32_dtype_probe

use ft_api::FrankenTorchSession;
use ft_core::{DType, ExecutionMode};

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let mut bugs = 0u32;

    macro_rules! un {
        ($name:literal, $m:ident, $vals:expr) => {{
            let t = s.tensor_variable_f32($vals, vec![4], false).unwrap();
            match s.$m(t) {
                Ok(r) => match s.tensor_dtype(r) {
                    Ok(DType::F32) => {}
                    Ok(d) => {
                        println!("{:<28} UPCAST -> {:?}  (torch keeps F32)", $name, d);
                        bugs += 1;
                    }
                    Err(e) => println!("{:<28} dtype-err {e:?}", $name),
                },
                Err(e) => {
                    println!("{:<28} ERR {e:?}", $name);
                    bugs += 1;
                }
            }
        }};
    }

    // positive-domain inputs (valid for log/sqrt/bessel/gamma/etc.)
    let pos = || vec![0.5f32, 1.0, 2.0, 3.0];
    // general inputs
    let genv = || vec![-1.0f32, 0.5, 1.0, 2.0];
    // (0,1) inputs (for ndtri/logit/atanh/erfinv)
    let unit = || vec![0.1f32, 0.3, 0.5, 0.9];

    un!("digamma", tensor_digamma, pos());
    un!("i0", tensor_i0, pos());
    un!("i1", tensor_i1, pos());
    un!("i0e", tensor_i0e, pos());
    un!("i1e", tensor_i1e, pos());
    un!("erf", tensor_erf, genv());
    un!("erfc", tensor_erfc, genv());
    un!("erfcx", tensor_erfcx, genv());
    un!("erfinv", tensor_erfinv, vec![-0.5f32, 0.0, 0.5, 0.9]);
    un!("ndtr", tensor_ndtr, genv());
    un!("ndtri", tensor_ndtri, unit());
    un!("lgamma", tensor_lgamma, pos());
    un!("sinc", tensor_sinc, genv());
    un!("expm1", tensor_expm1, genv());
    un!("log1p", tensor_log1p, pos());
    un!("exp2", tensor_exp2, genv());
    un!("log2", tensor_log2, pos());
    un!("log10", tensor_log10, pos());
    un!("special_bessel_j0", tensor_special_bessel_j0, pos());
    un!("special_bessel_j1", tensor_special_bessel_j1, pos());
    un!("special_bessel_y0", tensor_special_bessel_y0, pos());
    un!("special_modified_bessel_k0", tensor_special_modified_bessel_k0, pos());
    un!("gelu", tensor_gelu, genv());
    un!("silu", tensor_silu, genv());
    un!("mish", tensor_mish, genv());
    un!("softplus", tensor_softplus, genv());
    un!("softsign", tensor_softsign, genv());
    un!("reciprocal", tensor_reciprocal, genv());
    un!("sign", tensor_sign, genv());
    un!("trunc", tensor_trunc, genv());
    un!("frac", tensor_frac, genv());
    un!("nansum", tensor_nansum, genv());
    un!("nanmean", tensor_nanmean, genv());
    un!("prod", tensor_prod, pos());

    // batch 2: activation composites + remaining special fns
    un!("elu", tensor_elu, genv());
    un!("selu", tensor_selu, genv());
    un!("hardsigmoid", tensor_hardsigmoid, genv());
    un!("hardswish", tensor_hardswish, genv());
    un!("relu6", tensor_relu6, vec![-1.0f32, 2.0, 5.0, 8.0]);
    un!("tanhshrink", tensor_tanhshrink, genv());
    un!("logsigmoid", tensor_logsigmoid, genv());
    un!("hardtanh", tensor_hardtanh, genv());
    un!("special_entr", tensor_special_entr, unit());
    un!("special_bessel_j1", tensor_special_bessel_j1, pos());
    un!("special_bessel_y1", tensor_special_bessel_y1, pos());
    un!("special_modified_bessel_k1", tensor_special_modified_bessel_k1, pos());
    un!("special_scaled_modified_bessel_k0", tensor_special_scaled_modified_bessel_k0, pos());

    // parameterized (direct calls)
    macro_rules! un1 {
        ($name:literal, $vals:expr, $call:expr) => {{
            let t = s.tensor_variable_f32($vals, vec![4], false).unwrap();
            match $call(&mut s, t) {
                Ok(r) => match s.tensor_dtype(r) {
                    Ok(DType::F32) => {}
                    Ok(d) => { println!("{:<28} UPCAST -> {:?}", $name, d); bugs += 1; }
                    Err(e) => println!("{:<28} dtype-err {e:?}", $name),
                },
                Err(e) => { println!("{:<28} ERR {e:?}", $name); bugs += 1; }
            }
        }};
    }
    un1!("celu", genv(), |s: &mut FrankenTorchSession, t| s.tensor_celu(t, 1.0));
    un1!("softshrink", genv(), |s: &mut FrankenTorchSession, t| s.tensor_softshrink(t, 0.5));
    un1!("hardshrink", genv(), |s: &mut FrankenTorchSession, t| s.tensor_hardshrink(t, 0.5));
    un1!("logit", unit(), |s: &mut FrankenTorchSession, t| s.tensor_logit(t, None));
    un1!("polygamma", pos(), |s: &mut FrankenTorchSession, t| s.tensor_polygamma(1, t));
    un1!("multigammaln", vec![1.5f32, 2.0, 3.0, 4.0], |s: &mut FrankenTorchSession, t| s.tensor_multigammaln(t, 2));

    // batch 3: unary specials
    un!("special_airy_ai", tensor_special_airy_ai, genv());
    un!("special_spherical_bessel_j0", tensor_special_spherical_bessel_j0, pos());

    // batch 3: binary composites (both f32 -> f32 in torch)
    macro_rules! bin {
        ($name:literal, $a:expr, $b:expr, $call:expr) => {{
            let x = s.tensor_variable_f32($a, vec![4], false).unwrap();
            let y = s.tensor_variable_f32($b, vec![4], false).unwrap();
            match $call(&mut s, x, y) {
                Ok(r) => match s.tensor_dtype(r) {
                    Ok(DType::F32) => {}
                    Ok(d) => { println!("{:<28} UPCAST -> {:?}", $name, d); bugs += 1; }
                    Err(e) => println!("{:<28} dtype-err {e:?}", $name),
                },
                Err(e) => { println!("{:<28} ERR {e:?}", $name); bugs += 1; }
            }
        }};
    }
    let p4 = || vec![0.5f32, 1.0, 2.0, 3.0];
    bin!("xlogy", p4(), p4(), |s: &mut FrankenTorchSession, x, y| s.tensor_xlogy(x, y));
    bin!("xlog1py", p4(), p4(), |s: &mut FrankenTorchSession, x, y| s.tensor_xlog1py(x, y));
    bin!("special_zeta", vec![2.0f32, 3.0, 4.0, 5.0], p4(), |s: &mut FrankenTorchSession, x, y| s.tensor_special_zeta(x, y));
    bin!("rel_entr", p4(), p4(), |s: &mut FrankenTorchSession, x, y| s.tensor_rel_entr(x, y));
    bin!("special_gammainc", p4(), p4(), |s: &mut FrankenTorchSession, x, y| s.tensor_special_gammainc(x, y));
    bin!("special_gammaincc", p4(), p4(), |s: &mut FrankenTorchSession, x, y| s.tensor_special_gammaincc(x, y));

    // batch 4: matrix / distance ops (2D f32 inputs -> torch keeps f32)
    macro_rules! mat {
        ($name:literal, $build:expr) => {{
            match $build(&mut s) {
                Ok(r) => match s.tensor_dtype(r) {
                    Ok(DType::F32) => {}
                    Ok(d) => { println!("{:<28} UPCAST -> {:?}", $name, d); bugs += 1; }
                    Err(e) => println!("{:<28} dtype-err {e:?}", $name),
                },
                Err(e) => { println!("{:<28} ERR {e:?}", $name); bugs += 1; }
            }
        }};
    }
    // SPD 3x3 for the square-matrix ops.
    let spd = |s: &mut FrankenTorchSession| {
        s.tensor_variable_f32(vec![4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0], vec![3, 3], false).unwrap()
    };
    mat!("pinv", |s: &mut FrankenTorchSession| { let m = spd(s); s.tensor_pinv(m) });
    mat!("matrix_power", |s: &mut FrankenTorchSession| { let m = spd(s); s.tensor_matrix_power(m, 3) });
    mat!("matrix_exp", |s: &mut FrankenTorchSession| { let m = spd(s); s.tensor_matrix_exp(m) });
    mat!("logdet", |s: &mut FrankenTorchSession| { let m = spd(s); s.tensor_logdet(m) });
    mat!("trace", |s: &mut FrankenTorchSession| { let m = spd(s); s.tensor_trace(m) });
    mat!("cholesky_solve", |s: &mut FrankenTorchSession| {
        let b = s.tensor_variable_f32(vec![1.0, 2.0, 3.0], vec![3, 1], false).unwrap();
        let l = s.tensor_variable_f32(vec![2.0, 0.0, 0.0, 0.5, 1.6, 0.0, 0.0, 0.6, 1.3], vec![3, 3], false).unwrap();
        s.tensor_cholesky_solve(b, l, false)
    });
    mat!("cdist", |s: &mut FrankenTorchSession| {
        let x1 = s.tensor_variable_f32(vec![0.0, 0.0, 1.0, 1.0], vec![2, 2], false).unwrap();
        let x2 = s.tensor_variable_f32(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false).unwrap();
        s.tensor_cdist(x1, x2, 2.0)
    });
    mat!("pdist", |s: &mut FrankenTorchSession| {
        let x = s.tensor_variable_f32(vec![0.0, 0.0, 1.0, 1.0, 2.0, 0.0], vec![3, 2], false).unwrap();
        s.tensor_pdist(x, 2.0)
    });

    println!("\nf32 UPCAST/ERR bugs: {bugs}");
}
