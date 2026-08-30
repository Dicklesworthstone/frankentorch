//! Where should the streamed `dweight` gate go? — `frankentorch-hi9r6`.
//!
//! WHY A SWEEP AND NOT A SHIP. The streamed panel measured a PAIRED 1.244-1.254x against the
//! live incumbent at ONE shape (batch 8, 32 channels, 32x32, 3x3). A win at one shape is known
//! only at that shape: `conv3d-direct-gate-misset-w3pol` is the precedent where a fast path
//! validated at a single shape and gated on an open-ended `>=` pessimised everything above it by
//! 1.5-3.3x. The saving here is a panel ROUND TRIP, so it should scale with the panel and go
//! negative where the tile grid degenerates or the fork does not amortise. This finds where.
//!
//! THE ESTIMATOR. Both arms run INSIDE each rep, adjacent, so a drifting host hits them equally,
//! and the reported figure is min-of-reps per arm. The first rep of every shape is discarded:
//! `feedback_drift_gate_measures_sweep_length` records the first pass of a sweep running ~1.23x
//! slow for reasons that are not contention.
//!
//! ARM-INTERNAL. No incumbent, no ratio against PyTorch, no drift gate — this decides a gate
//! between two of OUR OWN paths, which is exactly the question a self-comparison answers. It is
//! NOT a standing and must never be quoted as one.
//!
//! Everything goes to STDERR so a remote runner returns it.

use std::time::Instant;

fn main() {
    let reps: usize = std::env::var("FT_REPS")
        .ok()
        .and_then(|t| t.trim().parse().ok())
        .unwrap_or(7);

    // batch, in_ch, out_ch, h, w, kh, kw, sh, sw
    let shapes: [(usize, usize, usize, usize, usize, usize, usize, usize, usize); 15] = [
        // THE SMALL END, sampled deliberately. `project_qr_panel_columnmajor` records the rule:
        // below the sizes anyone measured live the BATCHED-TINY planes, and an ungated inner
        // change has wrecked a shipped win there before. These are the shapes where the tile
        // grid degenerates to one task and the fork has nothing to amortise.
        (1, 1, 1, 8, 8, 3, 3, 1, 1),
        (2, 2, 4, 8, 8, 3, 3, 1, 1),
        (1, 3, 6, 12, 12, 3, 3, 1, 1),
        (64, 4, 4, 8, 8, 3, 3, 1, 1),
        (1, 16, 32, 7, 7, 3, 3, 1, 1),
        (1, 4, 8, 8, 8, 3, 3, 1, 1),
        (1, 8, 16, 16, 16, 3, 3, 1, 1),
        (2, 16, 16, 16, 16, 3, 3, 1, 1),
        (4, 16, 32, 32, 32, 3, 3, 1, 1),
        (8, 32, 32, 32, 32, 3, 3, 1, 1), // the scored lane shape
        (16, 32, 32, 32, 32, 3, 3, 1, 1),
        (8, 64, 32, 32, 32, 3, 3, 1, 1),
        (32, 16, 16, 16, 16, 3, 3, 1, 1),
        (8, 32, 32, 64, 64, 3, 3, 1, 1),
        (4, 32, 64, 32, 32, 3, 3, 2, 2),
    ];

    eprintln!(
        "DWEIGHT_GATE reps={reps} threads={} (both arms run inside each rep, adjacent; min of \
         reps after discarding the first)",
        rayon::current_num_threads()
    );

    for (batch, in_ch, out_ch, h, w, kh, kw, sh, sw) in shapes {
        let ph = h + 2;
        let pw = w + 2;
        let oh = (ph - kh) / sh + 1;
        let ow = (pw - kw) / sw + 1;
        let flat = batch * oh * ow;
        let patch_width = in_ch * kh * kw;
        let panel_mb = (flat * patch_width * 8) as f64 / (1024.0 * 1024.0);

        let padded: Vec<f64> = (0..batch * in_ch * ph * pw)
            .map(|i| ((i % 37) as f64) * 0.013 - 0.21)
            .collect();
        let weight_flat: Vec<f64> = (0..out_ch * in_ch * kh * kw)
            .map(|i| ((i % 11) as f64) * 0.0625 - 0.3125)
            .collect();
        let dout: Vec<f64> = (0..batch * out_ch * oh * ow)
            .map(|i| ((i % 23) as f64) * 0.019 - 0.19)
            .collect();

        let mut panel_ms = f64::INFINITY;
        let mut stream_ms = f64::INFINITY;
        let mut agree = true;
        for rep in 0..reps {
            let previous = ft_kernel_cpu::set_conv2d_dweight_streamed(false);
            let start = Instant::now();
            let (_, panel_dw, _) = ft_kernel_cpu::conv2d_backward_f64(
                &dout, &padded, &weight_flat, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw,
                out_ch, false,
            );
            let a = start.elapsed().as_secs_f64() * 1e3;

            ft_kernel_cpu::set_conv2d_dweight_streamed(true);
            let start = Instant::now();
            let (_, stream_dw, _) = ft_kernel_cpu::conv2d_backward_f64(
                &dout, &padded, &weight_flat, batch, in_ch, ph, pw, kh, kw, oh, ow, sh, sw,
                out_ch, false,
            );
            let b = start.elapsed().as_secs_f64() * 1e3;
            ft_kernel_cpu::set_conv2d_dweight_streamed(previous);

            // The two arms are bit-identical by construction; check it here too, because a
            // sweep that silently compared different numbers would report a meaningless ratio.
            if rep == 0 {
                agree = panel_dw.len() == stream_dw.len()
                    && panel_dw
                        .iter()
                        .zip(&stream_dw)
                        .all(|(x, y)| x.to_bits() == y.to_bits());
            }
            // Discard the first rep on BOTH arms: allocator and page-fault costs the rest of
            // the sweep does not pay.
            if rep > 0 {
                panel_ms = panel_ms.min(a);
                stream_ms = stream_ms.min(b);
            }
        }

        eprintln!(
            "DWEIGHT_GATE b={batch:>3} ci={in_ch:>3} co={out_ch:>3} {h:>3}x{w:<3} k{kh}x{kw} \
             s{sh}x{sw}  flat={flat:>7} pw={patch_width:>4} panel={panel_mb:8.2}MB  \
             panel={panel_ms:8.3}ms  streamed={stream_ms:8.3}ms  ratio={:6.3}x  bits={}",
            panel_ms / stream_ms,
            if agree { "identical" } else { "DIFFER" }
        );
    }
}
