//! Where does `slogdet` spend its time, on ONE estimator? — `frankentorch-x6wc3`.
//!
//! WHY THIS EXISTS, AND IT IS A CORRECTION TO MY OWN POSTED NUMBER. The h2h harness prints the LU
//! phase map from "one instrumented call" while the lane figure beside it is a MIN over 16 rounds.
//! I divided the first by the second and reported "slogdet is 64-78% LU". That mixes estimators
//! exactly the way NEGATIVE_EVIDENCE items 266 and 270 warn about, and the direction of the error
//! is not even obvious: a single call is usually SLOWER than a min-of-16, which would make the
//! true LU share LOWER than I quoted, but it depends on how cold that call is.
//!
//! Here every frame is min-of-N on the same host in the same process, so the split is a
//! decomposition rather than a quotient of two different statistics:
//!
//!   slogdet_contiguous_f64   the whole op
//!   lu_factor_contiguous_f64 the LU alone — `slogdet` is documented as returning sign and
//!                            logabsdet "from a single LU", so this is the shared frame
//!   difference               the slogdet-SPECIFIC residue: the sign/log-product pass
//!
//! plus `lu_stage_take_ns()` accumulated over the SAME calls, so the panel/solve/trailing split is
//! on the same population too.
//!
//! THE OVERLAP THIS DECIDES. `frankentorch-e1isq` owns the getrf panel. If slogdet is nearly all
//! LU, this bead has almost nothing of its own and should be handed over rather than worked in
//! parallel; if the residue is substantial, that residue is the only part that is mine.
//!
//! ARM-INTERNAL: no incumbent, no ratio, no drift gate, no PyTorch — runs on any rch worker.
//! Everything goes to STDERR so a remote runner returns it.

use ft_core::{DType, Device, TensorMeta};
use std::time::Instant;

fn main() {
    let n: usize = std::env::var("FT_N")
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(256);
    let reps: usize = std::env::var("FT_REPS")
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(9);

    // Diagonally dominant so the factorisation is well conditioned and pivoting does not dominate
    // by accident — the point is to time the common path, not a pathological one.
    let data: Vec<f64> = (0..n * n)
        .map(|idx| {
            let i = idx / n;
            let j = idx % n;
            let v = ((i * 31 + j * 17) % 23) as f64 * 0.05 - 0.5;
            if i == j { v + (n as f64) } else { v }
        })
        .collect();
    let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);

    eprintln!(
        "SLOGDET n={n} reps={reps} threads={} (arms adjacent inside each rep; min after \
         discarding the first)",
        rayon::current_num_threads()
    );

    let mut slog = f64::INFINITY;
    let mut lu = f64::INFINITY;
    let mut panel_ns = 0u64;
    let mut solve_ns = 0u64;
    let mut trail_ns = 0u64;
    let mut counted = 0u64;

    for rep in 0..reps {
        let start = Instant::now();
        let s = ft_kernel_cpu::slogdet_contiguous_f64(&data, &meta).expect("slogdet");
        let t_slog = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&s);

        // Drain whatever the slogdet call left, then attribute only the LU call's own phases.
        let _ = ft_kernel_cpu::lu_stage_take_ns();
        let start = Instant::now();
        let l = ft_kernel_cpu::lu_factor_contiguous_f64(&data, &meta).expect("lu_factor");
        let t_lu = start.elapsed().as_secs_f64() * 1e3;
        std::hint::black_box(&l);
        let (p, s_ns, tr) = ft_kernel_cpu::lu_stage_take_ns();

        if rep > 0 {
            slog = slog.min(t_slog);
            lu = lu.min(t_lu);
            panel_ns += p;
            solve_ns += s_ns;
            trail_ns += tr;
            counted += 1;
        }
    }

    let ms = |v: u64| v as f64 / 1e6 / counted.max(1) as f64;
    let residue = slog - lu;
    eprintln!("SLOGDET slogdet_contiguous_f64   {slog:8.3} ms");
    eprintln!("SLOGDET lu_factor_contiguous_f64 {lu:8.3} ms   ({:.0}% of slogdet)", 100.0 * lu / slog);
    eprintln!(
        "SLOGDET slogdet-SPECIFIC residue {residue:8.3} ms   ({:.0}% of slogdet)",
        100.0 * residue / slog
    );
    eprintln!(
        "SLOGDET   LU phases, mean over the SAME calls: panel {:.3} solve {:.3} trailing {:.3} ms",
        ms(panel_ns),
        ms(solve_ns),
        ms(trail_ns)
    );
    eprintln!(
        "SLOGDET   panel is {:.0}% of the LU and {:.0}% of the whole op — that share is \
         frankentorch-e1isq's territory.",
        100.0 * ms(panel_ns) / lu,
        100.0 * ms(panel_ns) / slog
    );
}
