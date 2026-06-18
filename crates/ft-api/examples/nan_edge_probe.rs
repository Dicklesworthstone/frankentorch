//! NaN/inf edge-case differential probe for sort/topk/maximum/minimum/clamp/
//! nan_to_num/amax/amin vs torch. Prints ft outputs; compare to torch goldens.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn p(name: &str, v: &[f64]) {
    let s: Vec<String> = v
        .iter()
        .map(|x| {
            if x.is_nan() {
                "nan".into()
            } else {
                format!("{x}")
            }
        })
        .collect();
    println!("{name}: [{}]", s.join(", "));
}

fn main() {
    let inf = f64::INFINITY;
    let nan = f64::NAN;
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);

    let x = s
        .tensor_variable(vec![3.0, nan, 1.0, inf, -inf, 2.0], vec![6], false)
        .unwrap();
    let (sv, si) = s.tensor_sort(x, 0, false).unwrap();
    p("sort_asc_v", &s.tensor_values(sv).unwrap());
    println!("sort_asc_i {si:?}");
    let (dv, di) = s.tensor_sort(x, 0, true).unwrap();
    p("sort_desc_v", &s.tensor_values(dv).unwrap());
    println!("sort_desc_i {di:?}");
    let (tv, ti) = s.tensor_topk(x, 3, 0, true, true).unwrap();
    p("topk_largest_v", &s.tensor_values(tv).unwrap());
    println!("topk_largest_i {ti:?}");

    let a = s
        .tensor_variable(vec![1.0, nan, 3.0, 2.0], vec![4], false)
        .unwrap();
    let b = s
        .tensor_variable(vec![nan, 5.0, 1.0, 2.0], vec![4], false)
        .unwrap();
    let mx = s.tensor_maximum(a, b).unwrap();
    p("maximum", &s.tensor_values(mx).unwrap());
    let mn = s.tensor_minimum(a, b).unwrap();
    p("minimum", &s.tensor_values(mn).unwrap());

    let c = s
        .tensor_variable(vec![nan, -5.0, 0.5, 10.0], vec![4], false)
        .unwrap();
    let cl = s.tensor_clamp(c, -1.0, 1.0).unwrap();
    p("clamp", &s.tensor_values(cl).unwrap());

    let d = s
        .tensor_variable(vec![nan, inf, -inf, 2.0], vec![4], false)
        .unwrap();
    let dn = s
        .tensor_nan_to_num(d, 0.0, Some(1e30), Some(-1e30))
        .unwrap();
    p("nan_to_num", &s.tensor_values(dn).unwrap());

    let x2 = s
        .tensor_variable(vec![1.0, nan, 3.0, 2.0], vec![2, 2], false)
        .unwrap();
    let amax = s.tensor_amax(x2, 1).unwrap();
    p("amax_dim1", &s.tensor_values(amax).unwrap());
    let amin = s.tensor_amin(x2, 1).unwrap();
    p("amin_dim1", &s.tensor_values(amin).unwrap());
}
