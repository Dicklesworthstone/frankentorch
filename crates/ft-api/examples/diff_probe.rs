//! Differential edge-case probe vs PyTorch (frankentorch Phase-B parity sweep).
//! Prints `name|v0,v1,...` lines for sign/zero/NaN/Inf-prone ops so a matching
//! torch script (scripts/diff_probe_torch.py) can be diffed against it.
//!
//!   cargo run -q -p ft-api --example diff_probe

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

const NAN: f64 = f64::NAN;
const INF: f64 = f64::INFINITY;

fn fmt(v: &[f64]) -> String {
    v.iter()
        .map(|&x| {
            if x.is_nan() {
                "nan".to_string()
            } else if x == f64::INFINITY {
                "inf".to_string()
            } else if x == f64::NEG_INFINITY {
                "-inf".to_string()
            } else {
                format!("{x:.17e}")
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let n = 12;
    let a = vec![
        -5.5, -3.0, 3.0, 5.5, -0.0, 0.0, -7.0, 2.5, NAN, INF, -INF, 1.0,
    ];
    let b = vec![
        2.0, 2.0, -2.0, 2.0, 1.0, -1.0, 2.0, -2.0, 1.0, 0.0, 1.0, 0.0,
    ];
    let av = s.tensor_variable(a.clone(), vec![n], false).unwrap();
    let bv = s.tensor_variable(b.clone(), vec![n], false).unwrap();
    let val = |s: &mut FrankenTorchSession, id| s.tensor_values(id).unwrap();

    macro_rules! bin {
        ($name:literal, $m:ident, $x:expr, $y:expr) => {{
            let r = s.$m($x, $y).unwrap();
            let v = val(&mut s, r);
            println!("{}|{}", $name, fmt(&v));
        }};
    }

    bin!("remainder", tensor_remainder, av, bv);
    bin!("fmod", tensor_fmod, av, bv);
    bin!("floor_divide", tensor_floor_divide, av, bv);
    bin!("copysign", tensor_copysign, av, bv);
    bin!("nextafter", tensor_nextafter, av, bv);
    bin!("hypot", tensor_hypot, av, bv);
    bin!("logaddexp", tensor_logaddexp, av, bv);
    bin!("fmax", tensor_fmax, av, bv);
    bin!("ldexp", tensor_ldexp, av, bv);

    // floor_divide: comprehensive sign/zero/inf/nan edge set (frankentorch-bh6bh).
    // torch uses aten div_floor_floating, NOT floor(a/b): ±inf dividend -> NaN,
    // -5/+inf -> -1, but inf/0 -> inf (b==0 short-circuit).
    let fa = s.tensor_variable(
        vec![
            1.0, -1.0, 0.0, INF, -INF, INF, -INF, INF, -INF, NAN, 1.0, INF, 5.0, -5.0, 7.0, -7.0,
            0.3, 2.5, -0.0, 6.5, -6.5,
        ],
        vec![21],
        false,
    ).unwrap();
    let fb = s.tensor_variable(
        vec![
            0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0, 0.0, 0.0, 1.0, NAN, INF, INF, INF, 2.0, 2.0, 0.1,
            0.5, 3.0, 2.0, 2.0,
        ],
        vec![21],
        false,
    ).unwrap();
    bin!("floor_divide_edge", tensor_floor_divide, fa, fb);

    // xlogy: x*log(y), with the x==0 short-circuit (0*log(0)=0, 0*log(-1)=0).
    let xx = s.tensor_variable(vec![0.0, 0.0, 2.0, 3.0, 0.5, 1.0], vec![6], false).unwrap();
    let yy = s.tensor_variable(vec![0.0, -1.0, 0.0, 2.0, 4.0, -3.0], vec![6], false).unwrap();
    bin!("xlogy", tensor_xlogy, xx, yy);

    // unary / scalar-arg
    let r = s.tensor_signbit(av).unwrap();
    println!("signbit|{}", fmt(&val(&mut s, r)));
    let c = s.tensor_variable(vec![0.0, 0.5, 1.0, -1.0, 2.0, -0.5], vec![6], false).unwrap();
    let r = s.tensor_sinc(c).unwrap();
    println!("sinc|{}", fmt(&val(&mut s, r)));
    let r = s.tensor_float_power(av, 0.5).unwrap();
    println!("float_power_0.5|{}", fmt(&val(&mut s, r)));
    let r = s.tensor_nan_to_num(av, 0.0, None, None).unwrap();
    println!("nan_to_num|{}", fmt(&val(&mut s, r)));
    let r = s.tensor_heaviside(av, bv).unwrap();
    println!("heaviside|{}", fmt(&val(&mut s, r)));

    // ── batch 2: NaN-propagation + domain edges ──────────────────────────
    // maximum/minimum PROPAGATE NaN (unlike fmax/fmin which ignore it).
    let ma = s.tensor_variable(vec![NAN, 1.0, NAN, -INF, INF, 3.0], vec![6], false).unwrap();
    let mb = s.tensor_variable(vec![1.0, NAN, NAN, 5.0, 5.0, -3.0], vec![6], false).unwrap();
    bin!("maximum", tensor_maximum, ma, mb);
    bin!("minimum", tensor_minimum, ma, mb);

    // atan2(y, x): quadrant/edge behavior.
    let ya = s.tensor_variable(vec![0.0, -0.0, 1.0, -1.0, INF, INF, 0.0, NAN], vec![8], false).unwrap();
    let xa = s.tensor_variable(vec![1.0, 1.0, 0.0, 0.0, INF, -INF, -1.0, 1.0], vec![8], false).unwrap();
    bin!("atan2", tensor_atan2, ya, xa);

    // clamp(x, 0, 1) incl NaN input; and clamp(x, 1, 0) (min>max).
    let cl = s.tensor_variable(vec![-1.0, 0.5, 2.0, NAN, INF, -INF], vec![6], false).unwrap();
    let r = s.tensor_clamp(cl, 0.0, 1.0).unwrap();
    println!("clamp01|{}", fmt(&val(&mut s, r)));
    let r = s.tensor_clamp(cl, 1.0, 0.0).unwrap();
    println!("clamp_minmax|{}", fmt(&val(&mut s, r)));

    // Unary domain edges.
    macro_rules! un {
        ($name:literal, $m:ident, $vec:expr) => {{
            let t = s.tensor_variable($vec, vec![6], false).unwrap();
            let r = s.$m(t).unwrap();
            println!("{}|{}", $name, fmt(&val(&mut s, r)));
        }};
    }
    un!("asin", tensor_asin, vec![2.0, -2.0, 1.0, -1.0, 0.0, NAN]);
    un!("acos", tensor_acos, vec![2.0, -2.0, 1.0, -1.0, 0.0, NAN]);
    un!("log", tensor_log, vec![-1.0, 0.0, 1.0, INF, -INF, NAN]);
    un!("sqrt", tensor_sqrt, vec![-1.0, 0.0, 4.0, INF, -INF, NAN]);
    un!("rsqrt", tensor_rsqrt, vec![0.0, 4.0, -1.0, INF, -0.0, NAN]);
    un!("log1p", tensor_log1p, vec![-2.0, -1.0, 0.0, INF, -0.5, NAN]);
    un!("expm1", tensor_expm1, vec![-INF, 0.0, INF, -1.0, 1.0, NAN]);
    un!("lgamma", tensor_lgamma, vec![0.0, -1.0, -2.0, 0.5, INF, -INF]);
    un!("digamma", tensor_digamma, vec![0.0, -1.0, -2.0, 0.5, 1.0, INF]);
    un!("erfinv", tensor_erfinv, vec![1.0, -1.0, 2.0, -2.0, 0.0, NAN]);
    let t = s.tensor_variable(vec![0.0, 1.0, 0.5, -0.1, 1.1, NAN], vec![6], false).unwrap();
    let r = s.tensor_logit(t, None).unwrap();
    println!("logit|{}", fmt(&val(&mut s, r)));

    // ── batch 3: activation extremes (overflow / inf / nan) ──────────────
    // x covering -inf, large negative, 0, large positive (overflow region),
    // +inf, nan — where naive exp/log compositions blow up vs torch.
    macro_rules! un8 {
        ($name:literal, $m:ident) => {{
            let t = s
                .tensor_variable(vec![-INF, -100.0, -1.0, 0.0, 1.0, 50.0, 100.0, INF, NAN], vec![9], false)
                .unwrap();
            let r = s.$m(t).unwrap();
            println!("{}|{}", $name, fmt(&val(&mut s, r)));
        }};
    }
    un8!("softplus", tensor_softplus);
    un8!("sigmoid", tensor_sigmoid);
    un8!("tanh", tensor_tanh);
    un8!("gelu", tensor_gelu);
    un8!("silu", tensor_silu);
    un8!("mish", tensor_mish);
    un8!("softsign", tensor_softsign);
    un8!("erf", tensor_erf);
    un8!("erfc", tensor_erfc);
    un8!("reciprocal", tensor_reciprocal);

    // ── batch 4: rounding (banker's) + exp/log at extremes / halfway ─────
    macro_rules! unr {
        ($name:literal, $m:ident) => {{
            let t = s
                .tensor_variable(
                    vec![-INF, INF, NAN, -0.0, 2.5, 3.5, -2.5, 0.5, -0.5, 1e20],
                    vec![10],
                    false,
                )
                .unwrap();
            let r = s.$m(t).unwrap();
            println!("{}|{}", $name, fmt(&val(&mut s, r)));
        }};
    }
    unr!("round", tensor_round);
    unr!("trunc", tensor_trunc);
    unr!("frac", tensor_frac);
    unr!("ceil", tensor_ceil);
    unr!("floor", tensor_floor);
    unr!("sign", tensor_sign);
    macro_rules! unl {
        ($name:literal, $m:ident) => {{
            let t = s
                .tensor_variable(vec![-INF, -1.0, 0.0, -0.0, 1.0, 2.0, 8.0, 710.0, INF, NAN], vec![10], false)
                .unwrap();
            let r = s.$m(t).unwrap();
            println!("{}|{}", $name, fmt(&val(&mut s, r)));
        }};
    }
    unl!("exp", tensor_exp);
    unl!("exp2", tensor_exp2);
    unl!("log2", tensor_log2);
    unl!("log10", tensor_log10);

    // ── batch 5: reductions over inf/nan + comparison NaN semantics ──────
    let mk = |s: &mut FrankenTorchSession, v: Vec<f64>| {
        let n = v.len();
        s.tensor_variable(v, vec![n], false).unwrap()
    };
    macro_rules! pr {
        ($name:literal, $id:expr) => {{
            let id = $id;
            println!("{}|{}", $name, fmt(&val(&mut s, id)));
        }};
    }
    let t = mk(&mut s, vec![2.0, 0.0, INF]);
    pr!("prod_0inf", s.tensor_prod(t).unwrap());
    let t = mk(&mut s, vec![INF, -INF]);
    pr!("sum_infninf", s.tensor_sum(t).unwrap());
    let t = mk(&mut s, vec![1.0, NAN, 2.0]);
    pr!("nansum", s.tensor_nansum(t).unwrap());
    let t = mk(&mut s, vec![NAN, NAN]);
    pr!("nansum_allnan", s.tensor_nansum(t).unwrap());
    let t = mk(&mut s, vec![1.0, NAN, 3.0]);
    pr!("nanmean", s.tensor_nanmean(t).unwrap());
    let t = mk(&mut s, vec![NAN, NAN]);
    pr!("nanmean_allnan", s.tensor_nanmean(t).unwrap());
    let t = mk(&mut s, vec![5.0]);
    pr!("var_n1_c1", s.tensor_var(t, 1).unwrap());
    let t = mk(&mut s, vec![5.0]);
    pr!("var_n1_c0", s.tensor_var(t, 0).unwrap());
    let t = mk(&mut s, vec![5.0]);
    pr!("std_n1_c1", s.tensor_std(t, 1).unwrap());
    let t = mk(&mut s, vec![1.0, NAN, 3.0]);
    pr!("amax_nan", s.tensor_amax(t, 0).unwrap());
    let t = mk(&mut s, vec![1.0, NAN, 3.0]);
    pr!("amin_nan", s.tensor_amin(t, 0).unwrap());
    let t = mk(&mut s, vec![1.0, 0.0, INF, 2.0]);
    pr!("cumprod_0inf", s.tensor_cumprod(t, 0).unwrap());
    let t = mk(&mut s, vec![1.0, INF, -INF, 1.0]);
    pr!("cumsum_infninf", s.tensor_cumsum(t, 0).unwrap());
    let t = mk(&mut s, vec![-INF, -INF]);
    pr!("logsumexp_ninf", s.tensor_logsumexp(t, 0).unwrap());
    let t = mk(&mut s, vec![INF, 1.0]);
    pr!("logsumexp_inf", s.tensor_logsumexp(t, 0).unwrap());
    let t = mk(&mut s, vec![INF, 1.0, 2.0]);
    pr!("norm2_inf", s.tensor_norm(t, 2.0).unwrap());
    let a = mk(&mut s, vec![NAN, 1.0, INF, -INF]);
    let b = mk(&mut s, vec![NAN, 1.0, INF, -INF]);
    pr!("isclose_eqnanF", s.tensor_isclose(a, b, 1e-5, 1e-8, false).unwrap());
    let a = mk(&mut s, vec![NAN, 1.0, INF, -INF]);
    let b = mk(&mut s, vec![NAN, 1.0, INF, -INF]);
    pr!("isclose_eqnanT", s.tensor_isclose(a, b, 1e-5, 1e-8, true).unwrap());

    // ── batch 6: special-function ACCURACY (Cephes vs torch, tight tol) ──
    // Positive in-domain points; flagging only gaps >> 1 ULP (real accuracy
    // bugs, like the digamma B10-truncation found earlier). i0/i1/i0e/i1e/erfcx/
    // ndtr/ndtri all match torch to ~1e-16.
    // CAUTION — bessel_j0/j1/y0 INTENTIONALLY diverge ~1e-8 from torch and that
    // is CORRECT: ft routes them through libm (f64-accurate, mpmath-verified to
    // ~16 digits), while torch.special.bessel_j0 is only ~7-8 digits (it promotes
    // single-precision Cephes polynomials). Do NOT "fix" j0/j1/y0 toward torch.
    // bessel_k0 is the reverse: ft's table-free series is ~1.5e-8 worst (x≈9) vs
    // torch's ~1-ULP Cephes (frankentorch-4ixyt). libm has no k0/k1.
    macro_rules! sp {
        ($name:literal, $m:ident) => {{
            let t = s
                .tensor_variable(vec![0.5, 1.0, 2.0, 3.0, 5.0, 8.0], vec![6], false)
                .unwrap();
            let r = s.$m(t).unwrap();
            println!("{}|{}", $name, fmt(&val(&mut s, r)));
        }};
    }
    sp!("i0", tensor_i0);
    sp!("i1", tensor_i1);
    sp!("i0e", tensor_i0e);
    sp!("i1e", tensor_i1e);
    sp!("bessel_j0", tensor_special_bessel_j0);
    sp!("bessel_j1", tensor_special_bessel_j1);
    sp!("bessel_y0", tensor_special_bessel_y0);
    sp!("bessel_k0", tensor_special_modified_bessel_k0);
    sp!("erfcx", tensor_erfcx);
    // ndtr over a centered range; ndtri over (0,1).
    let t = s.tensor_variable(vec![-3.0, -1.0, 0.0, 1.0, 2.0, 3.0], vec![6], false).unwrap();
    let r = s.tensor_ndtr(t).unwrap();
    println!("ndtr|{}", fmt(&val(&mut s, r)));
    let t = s.tensor_variable(vec![0.01, 0.1, 0.25, 0.5, 0.9, 0.99], vec![6], false).unwrap();
    let r = s.tensor_ndtri(t).unwrap();
    println!("ndtri|{}", fmt(&val(&mut s, r)));
}
