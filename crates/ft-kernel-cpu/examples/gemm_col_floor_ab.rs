//! `frankentorch-stale-tuning-constants-lzku6` lane 5 — MIN_BLOCK_COLS.
//!
//! # MECHANISM MODEL: ONE KNOB, FOUR ROLES
//!
//! `MIN_BLOCK_COLS = 128` is not one constant doing one job. It appears as:
//!
//!   1. a 2-D TILING ADMISSION GATE, `n >= 2 * MIN_BLOCK_COLS`, in six GEMM entries;
//!   2. a COLUMN-PARALLEL ADMISSION GATE, `n >= 4 * MIN_BLOCK_COLS`, in `should_parallelize_cols`;
//!   3. a FLOOR ON BLOCK WIDTH in `block_cols`, `n.div_ceil(threads).max(MIN_BLOCK_COLS)`;
//!   4. a FLOOR ON THE 2-D GRID's column dimension in `tile_shape`.
//!
//! **So moving the constant moves four things at once**, and a ladder over it would measure their
//! mixture — `feedback_one_knob_is_secretly_two` in a worse form than the case that named it,
//! where decoupling a 1-D `mb` sweep from thread occupancy turned a NULL into a shipped 1.5468x.
//!
//! # WHY LANE 4's FINDING DOES NOT TRANSFER
//!
//! Lane 4 concluded `HOUSEHOLDER_PANEL_WIDTH` has NO free axis: it is pinned by exact-match to a
//! 32 hard-coded in three other places, changes no computation, and can only switch a route off.
//! **This constant is the opposite pathology.** It genuinely changes the computation — tile widths
//! and therefore tile counts — so a free axis exists. What it lacks is SEPARABILITY. Lane 4's
//! answer was "there is nothing to sweep"; lane 5's is "there is something to sweep and it must be
//! swept one role at a time". Reusing lane 4's conclusion here would have skipped a real lever.
//!
//! # THE ONE ROLE WITH A CLEAN LEVER, ALREADY BUILT AND NEVER MEASURED
//!
//! Role 4 has a decoupled runtime toggle — `set_gemm_tile_col_floor_adaptive`, item 170 — which
//! lowers ONLY `tile_shape`'s floor (to `MIN_BLOCK_COLS_ADAPTIVE = 32`) and touches neither
//! admission gate nor `block_cols`. It ships OFF and its own doc says it is **UNBUILT**: "whether
//! relaxing the floor helps is NOT knowable without measuring, which is the whole reason this is a
//! toggle rather than an edit." That is lane 5's experiment, sitting there pre-decoupled.
//!
//! The sign is genuinely open, per that doc: more, narrower strips give the pool more to steal and
//! even out a ragged edge, but each strip re-reads the whole `A` panel, and
//! `project_gemm_bandwidth_vein` records this family as DRAM-bound in exactly that way.
//!
//! # CENSUS FIRST — WHERE DOES THE FLOOR ACTUALLY BIND?
//!
//! The floor binds only when the thread-aware split is narrower than it: `n.div_ceil(q) < 128`,
//! where `q` is the grid's column count. At 16 threads `tile_grid` gives `(p,q) = (4,4)`, so the
//! binding window is `n < 512` — and the 2-D path is only entered at `n >= 256`. **The lever can
//! therefore only matter for `256 <= n < 512`**, which the census measures rather than assumes,
//! via `gemm_tile_floor_census_take()` counting `(calls, floor_bound)` inside `tile_shape` itself.
//!
//! # THE NEGATIVE CONTROL IS THE POINT
//!
//! A shape with `n >= 512` does not bind the floor, so the toggle MUST show no effect there. That
//! arm is included deliberately: if it moves, the lever is not doing what the model says and no
//! reading from the binding shapes can be trusted. A lever without a negative control is a lever
//! you cannot distinguish from drift.
//!
//! Interleaved per rep, order reversed on odd reps, per-rep min, median of per-rep paired ratios,
//! both estimators, exact sign test, incumbent within-run spread, A/A arm — ledger 293/293a.
//!
//! BIT-EXACT either way: the toggle changes only M/N tiling, never K, so no sum is reassociated.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example gemm_col_floor_ab -- [reps]

mod interleaved;

/// `(label, in_ch, floor_binds_at_16_threads)`.
///
/// Driven through the REAL caller, not a synthetic GEMM: `conv2d_backward_masked_f64` with the
/// mask `[false, true, false]` computes dweight ONLY, and its GEMM is exactly the one the toggle's
/// doc names — `dgemm_tb(m=out_ch, k=flat, n=patch_width)`. At the scored conv2d shape
/// (batch 8, 34x34, k3x3, out_ch 32) that is `m=32, k=8192`, and `n = in_ch*kh*kw` is the axis
/// this lever lives on. `mod gemm` is private, so a synthetic call was not available anyway — and
/// the real op is the better instrument regardless, since a lever that only moves a hand-built
/// GEMM is `feedback_insitu_over_standalone`'s inverted ladder waiting to happen.
///
/// in_ch 32 -> n=288, 40 -> 360, 48 -> 432 all sit in the binding window 256 <= n < 512.
/// in_ch 128 -> n=1152 is the NEGATIVE CONTROL: above the window, so the floor cannot bind.
const SHAPES: [(&str, usize, bool); 4] = [
    ("dweight in_ch=32  n=288", 32, true),
    ("dweight in_ch=40  n=360", 40, true),
    ("dweight in_ch=48  n=432", 48, true),
    ("CONTROL in_ch=128 n=1152", 128, false),
];

const BATCH: usize = 8;
const OUT_CH: usize = 32;
const KH: usize = 3;
const KW: usize = 3;
const PH: usize = 34;
const PW: usize = 34;
const OH: usize = 32;
const OW: usize = 32;
const MASK: [bool; 3] = [false, true, false]; // dweight ONLY

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps = interleaved::reps_for(args.get(1).and_then(|v| v.parse().ok()).unwrap_or(12));

    interleaved::banner("MIN_BLOCK_COLS role 4 — 2-D tile column floor", reps);
    println!(
        "MECHANISM: MIN_BLOCK_COLS serves FOUR roles (two admission gates, block_cols' width \
         floor, tile_shape's grid floor). Sweeping the constant moves all four; this measures the \
         ONE role with a decoupled toggle, set_gemm_tile_col_floor_adaptive (item 170, UNBUILT)."
    );
    println!(
        "NEGATIVE CONTROL: n=1024 does not bind the floor at 16 threads, so the toggle MUST show \
         no effect there. If it moves, nothing else in this table is trustworthy."
    );

    for (label, in_ch, expect_bind) in SHAPES {
        let n = in_ch * KH * KW;
        let padded: Vec<f64> = (0..BATCH * in_ch * PH * PW)
            .map(|i| ((i % 37) as f64) * 0.013 - 0.21)
            .collect();
        let weight: Vec<f64> = (0..OUT_CH * in_ch * KH * KW)
            .map(|i| ((i % 11) as f64) * 0.0625 - 0.3125)
            .collect();
        // NON-UNIFORM dout, or the all-ones adjoint fast path replaces the GEMM under test.
        let dout: Vec<f64> = (0..BATCH * OUT_CH * OH * OW)
            .map(|i| ((i % 23) as f64) * 0.019 - 0.19)
            .collect();
        let run_dweight = || {
            ft_kernel_cpu::conv2d_backward_masked_f64(
                &dout, &padded, &weight, BATCH, in_ch, PH, PW, KH, KW, OH, OW, 1, 1, OUT_CH, MASK,
            )
        };

        // ---- CENSUS: counts, not timings, so no window is needed ----------------------
        ft_kernel_cpu::set_gemm_tile_col_floor_adaptive(false);
        let _ = ft_kernel_cpu::gemm_tile_floor_census_take();
        std::hint::black_box(run_dweight());
        let (calls, bound) = ft_kernel_cpu::gemm_tile_floor_census_take();
        println!(
            "\n{label}   CENSUS tile_shape calls={calls} floor_bound={bound} \
             (model says bind={expect_bind}; n={n})"
        );
        if (bound > 0) != expect_bind {
            println!(
                "  MODEL DISAGREES WITH THE CENSUS — the arithmetic above is wrong about this \
                 shape, and the A/B below cannot be read until that is resolved."
            );
        }
        if calls == 0 {
            println!("  this shape never reaches the 2-D tile grid — nothing for the lever to do");
            continue;
        }

        // ---- A/B: arm0 = floor SHIPPED (128), arm1 = floor ADAPTIVE (32), arm2 = A/A -----
        let times = interleaved::run(3, reps, 2, |i| {
            ft_kernel_cpu::set_gemm_tile_col_floor_adaptive(i == 1);
            let started = std::time::Instant::now();
            let out = run_dweight();
            let ms = started.elapsed().as_secs_f64() * 1_000.0;
            std::hint::black_box(&out);
            ms
        });
        ft_kernel_cpu::set_gemm_tile_col_floor_adaptive(false);

        // BIT-EXACTNESS, checked rather than taken from the doc comment.
        ft_kernel_cpu::set_gemm_tile_col_floor_adaptive(false);
        let (_, shipped, _) = run_dweight();
        ft_kernel_cpu::set_gemm_tile_col_floor_adaptive(true);
        let (_, adaptive, _) = run_dweight();
        ft_kernel_cpu::set_gemm_tile_col_floor_adaptive(false);
        let (shipped, adaptive) = (
            shipped.expect("dweight requested"),
            adaptive.expect("dweight requested"),
        );
        let differing = shipped
            .iter()
            .zip(&adaptive)
            .filter(|(x, y)| x.to_bits() != y.to_bits())
            .count();

        let (lo, hi, gate) = interleaved::spread(&times[0]);
        println!(
            "  incumbent (floor 128) WITHIN-RUN spread {lo:.4}-{hi:.4} ms = {gate:.3}x \
             (IQR {:.3}x) — an effect at or below {:.4} is UNRESOLVED",
            interleaved::iqr_ratio(&times[0]),
            gate - 1.0
        );
        println!(
            "  BIT-EXACT: {} differing elements of {}",
            differing,
            shipped.len()
        );
        println!("  {:>14} {:>10} {}", "arm", "median", interleaved::Verdict::header());
        for i in 0..3 {
            let arm = match i {
                0 => "floor 128",
                1 => "floor 32",
                _ => "A/A",
            };
            let trust = if i == 0 {
                format!("{:>9} {:>9} {:>8} {:>8}  incumbent", "-", "-", "-", "-")
            } else {
                interleaved::verdict(&times[0], &times[i], gate).row()
            };
            println!("  {arm:>14} {:>10.4} {trust}", interleaved::median(&times[i]));
        }
    }
    ft_kernel_cpu::set_gemm_tile_col_floor_adaptive(false);
    println!(
        "\nREADING. A PAIRED ratio above 1.0 means the ADAPTIVE floor (32) is faster than the \
         shipped 128. Read the CONTROL row first: it must be a null, or the lever is not isolated. \
         Then the all-cells rule across the binding shapes, and only then a paired lane \
         certification — this is a kernel result and ledger 291/292i are both kernel results that \
         did not survive one."
    );
}
