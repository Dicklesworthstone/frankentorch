// Probe a batch of structural/movement ops on f32 input: report OK / CRASH / dtype.
// Finds asymmetric-dtype crash bugs (F64-gated fast path + F64-only fall-through reader). cc.
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
fn main() {
    fn mk(s: &mut FrankenTorchSession, n: usize, shape: Vec<usize>) -> ft_autograd::TensorNodeId {
        let v: Vec<f32> = (0..n).map(|i| ((i % 97) as f32 - 48.0) * 0.01).collect();
        s.tensor_variable_f32(v, shape, false).unwrap()
    }
    macro_rules! probe {
        ($name:expr, $body:expr) => {{
            let r: Result<ft_core::DType, Box<dyn std::error::Error>> = (|| {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let o = $body(&mut s)?;
                Ok(s.tensor_dtype(o)?)
            })();
            match r {
                Ok(dt) => println!("{:<28} OK    dtype={:?}", $name, dt),
                Err(e) => println!("{:<28} CRASH {:?}", $name, e),
            }
        }};
    }

    probe!("take_along_dim", |s: &mut FrankenTorchSession| {
        let x = mk(s, 4096 * 64, vec![4096, 64]);
        let idx = s.tensor_variable(vec![0.0; 4096 * 8], vec![4096, 8], false)?;
        Ok::<_, Box<dyn std::error::Error>>(s.tensor_take_along_dim(x, idx, 1)?)
    });
    probe!("flip[0,1]", |s: &mut FrankenTorchSession| {
        let x = mk(s, 1024 * 1024, vec![1024, 1024]);
        Ok::<_, Box<dyn std::error::Error>>(s.tensor_flip(x, &[0, 1])?)
    });
    probe!("roll shift=5 dim=1", |s: &mut FrankenTorchSession| {
        let x = mk(s, 1024 * 1024, vec![1024, 1024]);
        Ok::<_, Box<dyn std::error::Error>>(s.tensor_roll(x, 5, 1)?)
    });
    probe!("tensordot", |s: &mut FrankenTorchSession| {
        let a = mk(s, 64 * 64, vec![64, 64]);
        let b = mk(s, 64 * 64, vec![64, 64]);
        Ok::<_, Box<dyn std::error::Error>>(s.tensor_tensordot(a, b, 1)?)
    });
    probe!("cartesian_prod", |s: &mut FrankenTorchSession| {
        let a = mk(s, 100, vec![100]);
        let b = mk(s, 100, vec![100]);
        Ok::<_, Box<dyn std::error::Error>>(s.tensor_cartesian_prod(&[a, b])?)
    });
    probe!("block_diag", |s: &mut FrankenTorchSession| {
        let a = mk(s, 64 * 64, vec![64, 64]);
        let b = mk(s, 64 * 64, vec![64, 64]);
        Ok::<_, Box<dyn std::error::Error>>(s.tensor_block_diag(&[a, b])?)
    });
    probe!("combinations r=2", |s: &mut FrankenTorchSession| {
        let a = mk(s, 500, vec![500]);
        Ok::<_, Box<dyn std::error::Error>>(s.tensor_combinations(a, 2, false)?)
    });
    probe!("trace", |s: &mut FrankenTorchSession| {
        let x = mk(s, 512 * 512, vec![512, 512]);
        Ok::<_, Box<dyn std::error::Error>>(s.tensor_trace(x)?)
    });
    probe!("diag", |s: &mut FrankenTorchSession| {
        let x = mk(s, 1024, vec![1024]);
        Ok::<_, Box<dyn std::error::Error>>(s.tensor_diag(x, 0)?)
    });
}
