//! `frankentorch-stale-tuning-constants-lzku6` lane 2 — WHICH lane actually reaches `lu_solve`?
//!
//! `lu_solve_contiguous_f64`'s blocked path, the one carrying `NB = 64`, is gated on
//! `num_rhs >= 64 && n >= 256`. Before re-tuning that constant it is worth knowing which board op
//! executes it, because this campaign has twice been redirected by exactly this question: the 2-D
//! trailing arm reached 5 of 31 calls on slogdet (ledger 291) and cholesky turned out never to call
//! `dgemm_sub_into` at all (292).
//!
//! Structurally slogdet is LU plus an O(n) diagonal log-product — no solve — while `inv` is LU plus
//! an O(n^3) getri tail that solves against the identity, so `num_rhs = n`. This checks that with
//! the counters instead of asserting it, because source reading has produced three confident wrong
//! answers on this codebase.
//!
//!   cargo run --release -p frankentorch-api --example lu_solve_census -- [n]

use ft_api::FrankenTorchSession;
use ft_core::ExecutionMode;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(512);

    let values: Vec<f64> = (0..n * n)
        .map(|idx| {
            let (i, j) = (idx / n, idx % n);
            let v = (((i * 7 + j * 13) % 101) as f64 - 50.0) / 25.0;
            if i == j { v + n as f64 } else { v }
        })
        .collect();

    println!("n={n}  (lu_solve blocked gate is num_rhs >= 64 && n >= 256)");

    for op in ["slogdet", "inv"] {
        let mut session = FrankenTorchSession::new(ExecutionMode::Strict);
        let x = session
            .tensor_variable(values.clone(), vec![n, n], false)
            .expect("leaf");
        let _ = ft_kernel_cpu::lu_solve_half_take_ns();
        match op {
            "slogdet" => {
                let (_s, l) = session.tensor_linalg_slogdet(x).expect("slogdet");
                std::hint::black_box(l);
            }
            _ => {
                let y = session.tensor_linalg_inv(x).expect("inv");
                std::hint::black_box(y);
            }
        }
        let (fwd, back) = ft_kernel_cpu::lu_solve_half_take_ns();
        println!(
            "  {op:<8} lu_solve halves: forward {:.4} ms, backward {:.4} ms  ->  {}",
            fwd as f64 / 1e6,
            back as f64 / 1e6,
            if fwd + back > 0 {
                "REACHES lu_solve"
            } else {
                "does NOT reach lu_solve"
            }
        );
    }
    println!(
        "\nREADING: the op that reaches it is the op that can certify a change to its NB. \
         Re-tuning a constant against a lane that never executes it would measure nothing, \
         which is ledger 292's finding restated."
    );
}
