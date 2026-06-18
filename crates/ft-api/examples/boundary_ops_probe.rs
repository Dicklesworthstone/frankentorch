//! Differential probe for boundary/offset ops vs torch (searchsorted, bucketize,
//! histc, diagonal, diag_embed, tril, triu, roll, cross, repeat_interleave).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let p = |s: &mut FrankenTorchSession, name: &str, id| {
        let v = s.tensor_values(id).unwrap();
        println!("{name}: {v:?}");
    };

    let ss = s.tensor_variable(vec![1.0, 3.0, 5.0, 7.0], vec![4], false).unwrap();
    let vals = s.tensor_variable(vec![3.0, 5.0, 0.0, 8.0, 4.0], vec![5], false).unwrap();
    let r = s.tensor_searchsorted(ss, vals, false).unwrap();
    p(&mut s, "searchsorted_left", r);
    let r = s.tensor_searchsorted(ss, vals, true).unwrap();
    p(&mut s, "searchsorted_right", r);

    let inp = s.tensor_variable(vec![0.0, 3.0, 5.0, 8.0, 4.0], vec![5], false).unwrap();
    let bnd = s.tensor_variable(vec![1.0, 3.0, 5.0, 7.0], vec![4], false).unwrap();
    let r = s.tensor_bucketize(inp, bnd, false).unwrap();
    p(&mut s, "bucketize_left", r);
    let r = s.tensor_bucketize(inp, bnd, true).unwrap();
    p(&mut s, "bucketize_right", r);

    let h = s.tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], false).unwrap();
    let r = s.tensor_histc(h, 4, 0.0, 5.0).unwrap();
    p(&mut s, "histc", r);

    let m = s.tensor_variable((1..=9).map(f64::from).collect(), vec![3, 3], false).unwrap();
    let r = s.tensor_diagonal(m, 1).unwrap();
    p(&mut s, "diag_off1", r);
    let r = s.tensor_diagonal(m, -1).unwrap();
    p(&mut s, "diag_offm1", r);
    let r = s.tensor_tril(m, 1).unwrap();
    p(&mut s, "tril_1", r);
    let r = s.tensor_triu(m, -1).unwrap();
    p(&mut s, "triu_m1", r);

    let de = s.tensor_variable(vec![1.0, 2.0], vec![2], false).unwrap();
    let r = s.tensor_diag_embed(de, 1).unwrap();
    p(&mut s, "diag_embed_off1", r);

    let v = s.tensor_variable(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5], false).unwrap();
    let r = s.tensor_roll(v, 2, 0).unwrap();
    p(&mut s, "roll_2", r);
    let r = s.tensor_roll(v, -1, 0).unwrap();
    p(&mut s, "roll_m1", r);

    let a = s.tensor_variable(vec![1.0, 0.0, 0.0], vec![3], false).unwrap();
    let b = s.tensor_variable(vec![0.0, 1.0, 0.0], vec![3], false).unwrap();
    let r = s.tensor_cross(a, b).unwrap();
    p(&mut s, "cross", r);

    let ri = s.tensor_variable(vec![1.0, 2.0, 3.0], vec![3], false).unwrap();
    let r = s.tensor_repeat_interleave(ri, 2).unwrap();
    p(&mut s, "repeat_interleave2", r);
}
