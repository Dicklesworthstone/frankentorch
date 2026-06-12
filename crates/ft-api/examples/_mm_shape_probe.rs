use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let x = s.tensor_variable(vec![1., 2., 3.], vec![3], false).unwrap();
    let reps = s.tensor_variable(vec![1., 2., 3.], vec![3], false).unwrap();
    let r = s.tensor_repeat_interleave_repeats(x, reps, None).unwrap();
    println!(
        "1d-tensor {:?} (torch [1,2,2,3,3,3])",
        s.tensor_values(r).unwrap()
    );
    let x = s.tensor_variable(vec![1., 2., 3.], vec![3], false).unwrap();
    let sc = s.tensor_variable(vec![2.], vec![1], false).unwrap();
    let r = s.tensor_repeat_interleave_repeats(x, sc, None).unwrap();
    println!(
        "scalar-tensor {:?} (torch [1,1,2,2,3,3])",
        s.tensor_values(r).unwrap()
    );
    let y = s
        .tensor_variable(vec![1., 2., 3., 4.], vec![2, 2], false)
        .unwrap();
    let reps = s.tensor_variable(vec![1., 2.], vec![2], false).unwrap();
    let r = s
        .tensor_repeat_interleave_repeats(y, reps, Some(0))
        .unwrap();
    println!(
        "dim0 {:?} (torch [[1,2],[3,4],[3,4]])",
        s.tensor_values(r).unwrap()
    );
    // negative -> error
    let x = s.tensor_variable(vec![1., 2.], vec![2], false).unwrap();
    let bad = s.tensor_variable(vec![1., -1.], vec![2], false).unwrap();
    println!(
        "neg-err {:?}",
        s.tensor_repeat_interleave_repeats(x, bad, None).is_err()
    );
}
