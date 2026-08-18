//! Does the 2-D GEMM tile grid starve conv2d's backward? — `frankentorch-hi9r6`, item 172.
//!
//! **UNBUILT**: written under a build freeze, never compiled or run. It exists so that the first
//! quiet window after the freeze answers item 170's question at BOTH pool widths in ONE invocation instead of four.
//!
//! WHAT IS OWED, AND WHY IT MUST BE ONE PROCESS
//!
//! Item 170 found conv2d's backward GEMMs get 6 uneven tiles on an 8-thread pool, because
//! `MIN_BLOCK_COLS` is a floor on block WIDTH and therefore a ceiling on block COUNT, and shipped
//! the fix as a default-off toggle rather than an edit — a peer's items 159/161/163 were reverted
//! by their item 164 for making that class of change on reasoning alone.
//!
//! Item 165c separately found that MY items 158/160 cut the FORWARD transpose's splittable units
//! from 256 to `batch`, and noted the key property both defects share: an under-subscription
//! penalty grows with pool width, while a memory-traffic penalty does not — so neither question
//! can be answered at one width. A peer's item 171 has since repaired the forward pass.
//!
//! SCOPE, STATED PRECISELY: this probe times `conv2d_backward_f64`, which contains the two GEMMs
//! item 170 is about and does NOT contain the forward transpose. It therefore answers item 170's
//! question ONLY. Item 171's fix needs its own forward-side probe, and it would be wrong to read
//! this one as covering it.
//!
//! Four separate runs would be four different host states, and pairing across runs is the error
//! this campaign has made three times (items 123/135/139 on pool width, 145 on pooled rows, 169 on
//! slot profiles). Everything here happens inside one process, on one ELF, in one window:
//!
//!   * `rayon::ThreadPoolBuilder` builds an explicit pool per width and `install`s the work in it,
//!     so `rayon::current_num_threads()` — which `tile_grid` reads to size the grid — sees that
//!     width. No environment variable, no second invocation.
//!   * `set_gemm_tile_col_floor_adaptive` is an `AtomicBool` precisely so it can be flipped
//!     between arms inside one process (item 25: a cross-binary comparison cannot attribute a few
//!     percent to any one change).
//!
//! ARM-INTERNAL ONLY. There is no incumbent here and no ratio against PyTorch, so nothing this
//! prints is a standing or a win. It answers "does the toggle move OUR kernel, and does the answer
//! depend on pool width" — a maintenance question whose answer decides whether the toggle is worth
//! putting in front of the h2h board at all.
//!
//! IN SITU, NOT A REPLICA. It times the real `conv2d_backward_f64` rather than a standalone GEMM
//! ladder. `feedback_insitu_over_standalone` records a standalone ladder INVERTING in situ — a
//! predicted 5.7x win measured as a 1.118x regression — because allocator warmth differed. The
//! GEMMs here are reached the way the lane reaches them.

use std::time::Instant;

// conv2d_masked's shape, the lane behind this bead's certified 5.73x standing.
const BATCH: usize = 8;
const IN_CH: usize = 32;
const OUT_CH: usize = 32;
const H: usize = 32;
const W: usize = 32;
const K: usize = 3;

const PH: usize = H + 2;
const PW: usize = W + 2;
const OH: usize = PH - K + 1;
const OW: usize = PW - K + 1;

const REPS: usize = 9;
const WIDTHS: [usize; 2] = [8, 64];

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .map(|raw| {
            raw.split_whitespace()
                .take(3)
                .collect::<Vec<_>>()
                .join(" / ")
        })
        .unwrap_or_else(|_| "unavailable".to_owned())
}

fn cpu_mhz() -> String {
    let mut mhz: Vec<f64> = (0..)
        .map_while(|cpu| {
            std::fs::read_to_string(format!(
                "/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq"
            ))
            .ok()
        })
        .filter_map(|raw| raw.trim().parse::<f64>().ok().map(|khz| khz / 1000.0))
        .collect();
    if mhz.is_empty() {
        return "unavailable".to_owned();
    }
    mhz.sort_by(|a, b| a.partial_cmp(b).unwrap());
    format!(
        "min={:.0} median={:.0} max={:.0} spread={:.2}x",
        mhz[0],
        mhz[mhz.len() / 2],
        mhz[mhz.len() - 1],
        mhz[mhz.len() - 1] / mhz[0]
    )
}

fn main() {
    let patch_width = IN_CH * K * K;
    let patch_count = OH * OW;
    let flat = BATCH * patch_count;

    let padded: Vec<f64> = (0..BATCH * IN_CH * PH * PW)
        .map(|i| ((i % 251) as f64) * 0.001 - 0.12)
        .collect();
    let weight: Vec<f64> = (0..OUT_CH * patch_width)
        .map(|i| ((i % 241) as f64) * 0.001 - 0.11)
        .collect();
    // NON-UNIFORM: an all-ones `dout` takes the 3x3 adjoint, which contains neither GEMM this
    // probe is about. The masked lane's upstream is non-uniform for the same reason.
    let dout: Vec<f64> = (0..BATCH * OUT_CH * patch_count)
        .map(|i| ((i % 197) as f64) * 0.0007 + 0.25)
        .collect();
    assert!(
        dout.iter().any(|v| v.to_bits() != 1.0f64.to_bits()),
        "dout must be non-uniform or this probes the fast path"
    );

    println!("gemm_tile_floor_probe (frankentorch-hi9r6 item 171)");
    println!("shape [{BATCH},{IN_CH},{H},{W}] k={K} s=1 pad=1 out_ch={OUT_CH}");
    println!(
        "the two GEMMs under test:\n  \
         dweight  dgemm_tb(m={OUT_CH}, k={flat}, n={patch_width})\n  \
         dpanel   dgemm   (m={flat}, k={OUT_CH}, n={patch_width})"
    );
    println!("pre  loadavg {}", loadavg());
    println!("pre  cpu_mhz {}", cpu_mhz());
    println!();
    println!(
        "{:>7}  {:>10}  {:>10}  {:>9}  {:>9}",
        "threads", "floor=128", "adaptive", "ratio", "tiles"
    );

    let mut checksums: Vec<(usize, bool, f64)> = Vec::new();

    for width in WIDTHS {
        let pool = match rayon::ThreadPoolBuilder::new().num_threads(width).build() {
            Ok(p) => p,
            Err(e) => {
                println!("{width:>7}  pool build failed: {e}");
                continue;
            }
        };

        // Both arms inside the SAME pool, alternating, so any drift in the window lands on both.
        let mut best = [f64::INFINITY; 2];
        for _ in 0..REPS {
            for (arm, adaptive) in [false, true].into_iter().enumerate() {
                ft_kernel_cpu::set_gemm_tile_col_floor_adaptive(adaptive);
                let started = Instant::now();
                let (dpadded, dweight, _) = pool.install(|| {
                    ft_kernel_cpu::conv2d_backward_f64(
                        &dout, &padded, &weight, BATCH, IN_CH, PH, PW, K, K, OH, OW, 1, 1, OUT_CH,
                        false,
                    )
                });
                let elapsed = started.elapsed().as_secs_f64() * 1_000.0;
                best[arm] = best[arm].min(elapsed);
                // A checksum per arm: the toggle is claimed BIT-EXACT, so these must match
                // exactly. Recorded rather than asserted so a mismatch is reported with the
                // timings rather than aborting the run that would explain it.
                let sum: f64 = dpadded.iter().map(|v| v.abs()).sum::<f64>()
                    + dweight.iter().map(|v| v.abs()).sum::<f64>();
                checksums.push((width, adaptive, sum));
            }
        }
        ft_kernel_cpu::set_gemm_tile_col_floor_adaptive(false);

        // What the grid should look like at this width, by item 170's arithmetic, so the timing
        // can be read against the scheduling change it is supposed to come from.
        let lim = (width as f64).sqrt().floor().max(1.0) as usize;
        let p = (1..=lim)
            .filter(|c| width.is_multiple_of(*c))
            .max()
            .unwrap_or(1);
        let q = width / p;
        let strips = |nb: usize| patch_width.div_ceil(nb);
        let rows_dw = OUT_CH.div_ceil(OUT_CH.div_ceil(p).max(8));
        println!(
            "{width:>7}  {:>10.3}  {:>10.3}  {:>8.3}x  {} -> {}",
            best[0],
            best[1],
            best[0] / best[1],
            rows_dw * strips(patch_width.div_ceil(q).max(128)),
            rows_dw * strips(patch_width.div_ceil(q).max(32))
        );
    }

    println!();
    let mismatch = !checksums.is_empty()
        && checksums
            .iter()
            .any(|(_, _, s)| (s - checksums[0].2).abs() > 0.0);
    println!(
        "bit-exactness across ALL arms and widths: {}",
        if mismatch {
            "*** MISMATCH — the toggle changed a VALUE; item 170 is void and this is a BUG ***"
        } else {
            "identical checksums (necessary, not sufficient — the unit test compares by to_bits)"
        }
    );
    println!(
        "READING IT: item 170 predicts the adaptive arm helps MORE at 64 threads than at 8, \
         because under-subscription scales with pool width while the extra A-panel traffic does \
         not. If the ratio is flat across widths, the tile count was not the binding constraint. \
         If the adaptive arm is SLOWER, the A-traffic dominates and item 170's toggle should stay \
         off — which is a result worth having, not a failure."
    );
    println!("post loadavg {}", loadavg());
    println!("post cpu_mhz {}", cpu_mhz());
}
