//! geqrf's `nb`/`leaf` ladder, re-swept BELOW the shipped value after the layout change.
//! `frankentorch-geqrf-misses-blocked-kernel-1zp6r`.
//!
//! WHY RE-SWEEP. `nb` trades panel cost (rises with nb) against how many trailing updates are
//! paid (falls with nb). The column-major forward pass (7c40b137) made the trailing update much
//! cheaper, so the optimum must move — and it moves DOWNWARD. The in-tree ladder's grid started
//! at the shipped nb=32 and could only ever confirm it; this one opens the low end.
//!
//! WHY AN EXAMPLE AND NOT THE TEST. The `#[test]` form rebuilds the whole test cfg of a 2.8 MB
//! lib and buffers its output until the run ends, which on a contended fleet means a 40-minute
//! silence that a retry then discards. An example links against the already-built lib and streams.
//!
//! Everything goes to STDERR so a remote runner returns it.

use ft_kernel_cpu::QrStageTimings;

fn main() {
    let n: usize = std::env::var("FT_N")
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(1024);
    let reps: usize = std::env::var("FT_REPS")
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(7);
    let nbs: Vec<usize> = std::env::var("FT_NBS")
        .unwrap_or_else(|_| "8,16,24,32,48,64".to_string())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();
    let leaves: Vec<usize> = std::env::var("FT_LEAVES")
        .unwrap_or_else(|_| "1,2,4".to_string())
        .split(',')
        .filter_map(|t| t.trim().parse().ok())
        .collect();

    let a: Vec<f64> = (0..n * n)
        .map(|idx| {
            let i = (idx / n) as f64;
            let j = (idx % n) as f64;
            ((i * 0.37).sin() + (j * 0.11).cos()) * 1.5 + if idx % (n + 1) == 0 { 4.0 } else { 0.0 }
        })
        .collect();

    eprintln!(
        "GEQRF_LADDER n={n} reps={reps} threads={} (interleaved: the whole grid runs inside each \
         round, so a drifting machine hits every cell equally)",
        rayon::current_num_threads()
    );
    // Warm up: the first call pays allocator and page-fault costs the rest do not.
    let _ = ft_kernel_cpu::geqrf_blocked_nb_f64(&a, n, n, 32, 2, None);

    let mut best: std::collections::BTreeMap<(usize, usize), (f64, QrStageTimings)> =
        std::collections::BTreeMap::new();
    for _rep in 0..reps {
        for &nb in &nbs {
            for &leaf in &leaves {
                let mut t = QrStageTimings::default();
                let start = std::time::Instant::now();
                let (_p, _tau) = ft_kernel_cpu::geqrf_blocked_nb_f64(&a, n, n, nb, leaf, Some(&mut t));
                let ms = start.elapsed().as_secs_f64() * 1e3;
                match best.get(&(nb, leaf)) {
                    Some((prev, _)) if *prev <= ms => {}
                    _ => {
                        best.insert((nb, leaf), (ms, t));
                    }
                }
            }
        }
    }
    let winner = best
        .iter()
        .min_by(|a, b| a.1.0.total_cmp(&b.1.0))
        .map(|(k, v)| (*k, v.0));
    for ((nb, leaf), (wall_ms, t)) in &best {
        let wall_ns = wall_ms * 1e6;
        let pct = |v: u128| 100.0 * v as f64 / wall_ns;
        let mid_ns = t
            .trailing_r_ns
            .saturating_sub(t.trailing_pack_ns + t.trailing_gemm_ns);
        eprintln!(
            "GEQRF_LADDER n={n:>5} nb={nb:>3} leaf={leaf:>2} wall={wall_ms:8.3}ms  \
             panel+T={:5.1}%  trailing_R={:5.1}%  pack={:5.1}% ({:7.3}ms)  subGEMM={:5.1}%  \
             midGEMM={:5.1}%",
            pct(t.panel_and_t_ns),
            pct(t.trailing_r_ns),
            pct(t.trailing_pack_ns),
            t.trailing_pack_ns as f64 / 1e6,
            pct(t.trailing_gemm_ns),
            pct(mid_ns),
        );
        // The panel+T bucket, split three ways. `panel_and_t_ns` has been the largest single
        // term since the column-major trailing update landed, and it wraps three different
        // things: the recursive dgeqrt3, one V transpose, and the outer dlarft T build.
        eprintln!(
            "GEQRF_LADDER n={n:>5} nb={nb:>3} leaf={leaf:>2}   PANEL SPLIT  factor={:5.1}% \
             ({:7.3}ms)  vT={:5.1}% ({:7.3}ms)  Tbuild={:5.1}% ({:7.3}ms)",
            pct(t.panel_factor_ns),
            t.panel_factor_ns as f64 / 1e6,
            pct(t.panel_v_transpose_ns),
            t.panel_v_transpose_ns as f64 / 1e6,
            pct(t.panel_t_build_ns),
            t.panel_t_build_ns as f64 / 1e6,
        );
    }
    if let Some(((nb, leaf), ms)) = winner {
        let shipped = best.get(&(32, 2)).map(|v| v.0);
        match shipped {
            Some(s) => eprintln!(
                "GEQRF_LADDER n={n:>5} WINNER nb={nb} leaf={leaf} at {ms:.3}ms; shipped nb=32 \
                 leaf=2 at {s:.3}ms -> {:.3}x",
                s / ms
            ),
            None => eprintln!("GEQRF_LADDER n={n:>5} WINNER nb={nb} leaf={leaf} at {ms:.3}ms"),
        }
    }
}
