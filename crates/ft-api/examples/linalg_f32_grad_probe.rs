use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    // well-conditioned 2x2 f32 matrix with grad
    let a =
        s.tensor_variable_f32(vec![2.0, 0.5, 0.3, 1.5], vec![2, 2], true).unwrap();
    let bvec = s.tensor_variable_f32(vec![1.0, 2.0], vec![2, 1], true).unwrap();

    macro_rules! p {
        ($name:expr, $call:expr) => {{
            match $call {
                Ok(t) => {
                    let dt = s.tensor_dtype(t).unwrap();
                    let loss = s.tensor_sum(t).unwrap();
                    match s.tensor_backward(loss) {
                        Ok(_) => {
                            let g = s.tensor_grad(a).unwrap();
                            println!(
                                "{:<10} OK   dtype={:?}  grad={}",
                                $name,
                                dt,
                                if g.is_some() { "flows" } else { "SEVERED" }
                            );
                        }
                        Err(e) => println!("{:<10} OK-fwd but BWD ERR {:?}", $name, e),
                    }
                }
                Err(e) => println!("{:<10} ERR  {:?}", $name, e),
            }
        }};
    }
    p!("det", s.tensor_linalg_det(a));
    p!("inv", s.tensor_linalg_inv(a));
    p!("solve", s.tensor_linalg_solve(a, bvec));
}
