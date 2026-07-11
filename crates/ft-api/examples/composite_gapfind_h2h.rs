use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn main() {
    let tag = std::env::var("FT_TAG").unwrap_or_else(|_| "FT".into());
    let v4000: Vec<f64> = (0..4000).map(|i| (i as f64) * 1e-3 + 0.5).collect();
    let m500: Vec<f64> = (0..250_000).map(|i| ((i % 997) as f64) * 1e-3).collect();
    let m128: Vec<f64> = (0..128 * 128)
        .map(|i| ((i % 131) as f64) * 1e-3 + 0.1)
        .collect();
    let v4m: Vec<f64> = (0..4_000_000).map(|i| ((i % 9973) as f64) * 1e-4).collect();

    // Each closure builds its inputs in the session, then times ONLY the op (inputs
    // materialized BEFORE Instant::now() — otherwise the 32MB tensor_variable copies swamp it).
    macro_rules! bench {
        ($name:expr, $setup:expr, $op:expr) => {{
            let mut best = f64::INFINITY;
            for _ in 0..6 {
                let mut s = FrankenTorchSession::new(ExecutionMode::Strict);
                let inp = $setup(&mut s);
                let t0 = Instant::now();
                let _y = $op(&mut s, &inp);
                let ms = t0.elapsed().as_secs_f64() * 1e3;
                if ms < best {
                    best = ms;
                }
                std::hint::black_box(&s);
            }
            println!("[{tag}] {:<16}: {best:.2} ms", $name);
        }};
    }

    bench!(
        "outer[4000]",
        |s: &mut FrankenTorchSession| (
            s.tensor_variable(v4000.clone(), vec![4000], false).unwrap(),
            s.tensor_variable(v4000.clone(), vec![4000], false).unwrap()
        ),
        |s: &mut FrankenTorchSession, i: &(_, _)| s.tensor_outer(i.0, i.1).unwrap()
    );
    bench!(
        "tensordot500d1",
        |s: &mut FrankenTorchSession| (
            s.tensor_variable(m500.clone(), vec![500, 500], false)
                .unwrap(),
            s.tensor_variable(m500.clone(), vec![500, 500], false)
                .unwrap()
        ),
        |s: &mut FrankenTorchSession, i: &(_, _)| s.tensor_tensordot(i.0, i.1, 1).unwrap()
    );
    bench!(
        "block_diag50x128",
        |s: &mut FrankenTorchSession| (0..50)
            .map(|_| s
                .tensor_variable(m128.clone(), vec![128, 128], false)
                .unwrap())
            .collect::<Vec<_>>(),
        |s: &mut FrankenTorchSession, ids: &Vec<_>| s.tensor_block_diag(ids).unwrap()
    );
    bench!(
        "inner4m",
        |s: &mut FrankenTorchSession| (
            s.tensor_variable(v4m.clone(), vec![4_000_000], false)
                .unwrap(),
            s.tensor_variable(v4m.clone(), vec![4_000_000], false)
                .unwrap()
        ),
        |s: &mut FrankenTorchSession, i: &(_, _)| s.tensor_inner(i.0, i.1).unwrap()
    );
    bench!(
        "dot4m",
        |s: &mut FrankenTorchSession| (
            s.tensor_variable(v4m.clone(), vec![4_000_000], false)
                .unwrap(),
            s.tensor_variable(v4m.clone(), vec![4_000_000], false)
                .unwrap()
        ),
        |s: &mut FrankenTorchSession, i: &(_, _)| s.tensor_dot(i.0, i.1).unwrap()
    );
}
