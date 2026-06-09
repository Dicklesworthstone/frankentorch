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
}
