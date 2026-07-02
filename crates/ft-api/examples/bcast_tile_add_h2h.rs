// Same-process A/B for the fused broadcast-tile binop (attention additive-bias pattern).
// FT_ORIG unset -> fused try_bcast_tile_binop; FT_ORIG set -> original broadcast_to + tape op.
// Inputs are materialized BEFORE Instant::now() (input-in-timed-region trap avoidance).
use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;
use std::time::Instant;

fn bench<F: FnMut() -> u128>(iters: usize, mut f: F) -> u128 {
    let mut best = u128::MAX;
    for _ in 0..iters {
        let t = f();
        if t < best {
            best = t;
        }
    }
    best
}

fn main() {
    let orig = std::env::var("FT_ORIG").is_ok();
    let tag = if orig { "ORIG(broadcast+tape)" } else { "FUSED" };
    // [B,H,S,S] scores + [1,1,S,S] causal-style bias tile (the ALiBi/relpos/additive-mask pattern).
    let (b, h, s) = (16usize, 16usize, 128usize); // n = 16*16*128*128 = 4,194,304
    let n = b * h * s * s;
    let sn = s * s;
    let scores_f64: Vec<f64> = (0..n).map(|i| (i as f64 % 1000.0) * 0.01 - 5.0).collect();
    let bias_f64: Vec<f64> = (0..sn).map(|i| (i as f64 % 97.0) * 0.02 - 1.0).collect();
    let scores_f32: Vec<f32> = scores_f64.iter().map(|&x| x as f32).collect();
    let bias_f32: Vec<f32> = bias_f64.iter().map(|&x| x as f32).collect();

    // f64
    {
        let mut ses = FrankenTorchSession::new(ExecutionMode::Strict);
        let scores = ses.tensor_variable(scores_f64.clone(), vec![b, h, s, s], false).unwrap();
        let bias = ses.tensor_variable(bias_f64.clone(), vec![1, 1, s, s], false).unwrap();
        // warm
        let _ = ses.tensor_add(scores, bias).unwrap();
        let t = bench(9, || {
            let t0 = Instant::now();
            let out = ses.tensor_add(scores, bias).unwrap();
            let e = t0.elapsed().as_micros();
            std::hint::black_box(out);
            e
        });
        println!("[{tag}] add f64 [16,16,128,128]+[1,1,128,128]: {:.2} ms", t as f64 / 1000.0);
    }
    // f32
    {
        let mut ses = FrankenTorchSession::new(ExecutionMode::Strict);
        let scores = ses.tensor_variable_f32(scores_f32.clone(), vec![b, h, s, s], false).unwrap();
        let bias = ses.tensor_variable_f32(bias_f32.clone(), vec![1, 1, s, s], false).unwrap();
        let _ = ses.tensor_add(scores, bias).unwrap();
        let t = bench(9, || {
            let t0 = Instant::now();
            let out = ses.tensor_add(scores, bias).unwrap();
            let e = t0.elapsed().as_micros();
            std::hint::black_box(out);
            e
        });
        println!("[{tag}] add f32 [16,16,128,128]+[1,1,128,128]: {:.2} ms", t as f64 / 1000.0);
    }
    // key-padding [B,1,1,S] add f32
    {
        let mut ses = FrankenTorchSession::new(ExecutionMode::Strict);
        let bias_kp: Vec<f32> = (0..b * s).map(|i| (i as f32 % 13.0) * 0.1 - 0.5).collect();
        let scores = ses.tensor_variable_f32(scores_f32.clone(), vec![b, h, s, s], false).unwrap();
        let bias = ses.tensor_variable_f32(bias_kp.clone(), vec![b, 1, 1, s], false).unwrap();
        let _ = ses.tensor_add(scores, bias).unwrap();
        let t = bench(9, || {
            let t0 = Instant::now();
            let out = ses.tensor_add(scores, bias).unwrap();
            let e = t0.elapsed().as_micros();
            std::hint::black_box(out);
            e
        });
        println!("[{tag}] add f32 keypad [16,16,128,128]+[16,1,1,128]: {:.2} ms", t as f64 / 1000.0);
    }
}
