use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
    let a = s.tensor_variable_f32(vec![1., 2., 3., 4.], vec![2, 2], false).unwrap();
    let b = s.tensor_variable_f32(vec![5., 6., 7., 8.], vec![2, 2], false).unwrap();
    macro_rules! p {
        ($eq:expr, $t:expr) => {
            match s.tensor_einsum($eq, $t) {
                Ok(t) => println!("{:<14} OK   dtype={:?}", $eq, s.tensor_dtype(t).unwrap()),
                Err(e) => println!("{:<14} ERR  {:?}", $eq, e),
            }
        };
    }
    p!("ij,jk->ik", &[a, b]);
    p!("ij->ji", &[a]);
    let c = s.tensor_variable_f32(vec![1.; 24], vec![2, 3, 4], false).unwrap();
    let d = s.tensor_variable_f32(vec![1.; 40], vec![2, 4, 5], false).unwrap();
    p!("bij,bjk->bik", &[c, d]);
    p!("ij,ij->", &[a, b]);
}
