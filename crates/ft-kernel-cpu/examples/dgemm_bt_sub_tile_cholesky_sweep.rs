//! `frankentorch-stale-tuning-constants-lzku6` — isolated Cholesky gate for the
//! `dgemm_bt_sub_into` 2-D tile arm.
//!
//! The candidate only changes scheduling of Cholesky's trailing `C -= A·A^T` GEMM. K is never
//! tiled, so each output element retains the incumbent `dgemm_mm` reduction, while M/N tiles are
//! disjoint C rectangles. The kernel test proves those bits, including a strided C window; this
//! runner proves the full Cholesky path actually takes the candidate arm before timing it.
//!
//! Arm 0 is the shipped default (tile toggle OFF), arm 1 enables the candidate, and arm 2 is the
//! shipped A/A null. `interleaved::run` owns the repetition order, paired estimator, exact sign
//! test, and incumbent-spread verdict rules from ledger 293.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example dgemm_bt_sub_tile_cholesky_sweep -- [reps]

mod interleaved;

use ft_core::{DType, Device, TensorMeta};

fn spd(n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..i {
            let v = ((i * 17 + j * 31) % 23) as f64 * 0.01 - 0.1;
            a[i * n + j] = v;
            a[j * n + i] = v;
        }
        a[i * n + i] = n as f64;
    }
    a
}

/// Fastest sample: `(wall_ms, panel_ns, trsm_ns, trailing_ns, zero_ns, tiled, column)`.
type Sample = (f64, u64, u64, u64, u64, u64, u64);

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps = interleaved::reps_for(args.get(1).and_then(|v| v.parse().ok()).unwrap_or(10));
    interleaved::banner("dgemm_bt_sub_into 2-D tile in blocked f64 Cholesky", reps);
    println!(
        "INCUMBENT: dgemm_bt_sub_into column split, tile toggle OFF (the shipped default). \\
         CANDIDATE: same dgemm_mm calls with K whole, partitioned only into disjoint M/N tiles."
    );
    println!(
        "SENTINEL: each fastest sample prints the dgemm_bt_sub_into tile/column split. A candidate \\
         with zero tile hits aborts; any residual column calls remain visible rather than being \\
         silently counted as candidate work."
    );

    for n in [256usize, 512, 1024] {
        let a = spd(n);
        let meta = TensorMeta::from_shape(vec![n, n], DType::F64, Device::Cpu);
        let mut fastest: Vec<Sample> = vec![(f64::INFINITY, 0, 0, 0, 0, 0, 0); 3];
        let times = interleaved::run(3, reps, 2, |arm| {
            let candidate = arm == 1;
            ft_kernel_cpu::set_dgemm_bt_sub_tile_2d(candidate);
            let _ = ft_kernel_cpu::cholesky_stage_take_ns();
            let _ = ft_kernel_cpu::dgemm_bt_sub_arm_hits_take();
            let started = std::time::Instant::now();
            let factor = ft_kernel_cpu::cholesky_contiguous_f64(&a, &meta, false).expect("chol");
            let wall = started.elapsed().as_secs_f64() * 1_000.0;
            std::hint::black_box(&factor);
            let (panel, trsm, trailing, zero) = ft_kernel_cpu::cholesky_stage_take_ns();
            let (tiled, column) = ft_kernel_cpu::dgemm_bt_sub_arm_hits_take();
            if candidate {
                assert!(tiled > 0, "n={n}: candidate arm never reached a 2-D tile");
            } else {
                assert_eq!(tiled, 0, "n={n}: shipped arm unexpectedly tiled");
                assert!(column > 0, "n={n}: shipped arm made no trailing calls");
            }
            if wall < fastest[arm].0 {
                fastest[arm] = (wall, panel, trsm, trailing, zero, tiled, column);
            }
            wall
        });
        ft_kernel_cpu::set_dgemm_bt_sub_tile_2d(false);

        let (lo, hi, gate) = interleaved::spread(&times[0]);
        println!(
            "\nn={n}  incumbent WITHIN-RUN spread {lo:.4}-{hi:.4} ms = {gate:.3}x (IQR {:.3}x)",
            interleaved::iqr_ratio(&times[0])
        );
        println!(
            "  {:>19} {:>9} {:>9} {:>9} {:>9} {:>9} {:>7} {:>7} {:>7} {}",
            "arm",
            "median",
            "panel",
            "TRSM",
            "trail",
            "zero",
            "tile",
            "column",
            "admit",
            interleaved::Verdict::header(),
        );
        for arm in 0..3 {
            let (_, panel, trsm, trailing, zero, tiled, column) = fastest[arm];
            let label = match arm {
                0 => "shipped(tile=OFF)",
                1 => "bt-2d(tile=ON)",
                _ => "A/A(tile=OFF)",
            };
            let trust = if arm == 0 {
                format!("{:>9} {:>9} {:>8} {:>8}  incumbent", "-", "-", "-", "-")
            } else {
                interleaved::verdict(&times[0], &times[arm], gate).row()
            };
            let ms = |v: u64| v as f64 / 1e6;
            let calls = tiled + column;
            let admitted = if calls == 0 {
                "-".to_owned()
            } else {
                format!("{:.0}%", 100.0 * tiled as f64 / calls as f64)
            };
            println!(
                "  {:>19} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>9.4} {:>7} {:>7} {:>7} {trust}",
                label,
                interleaved::median(&times[arm]),
                ms(panel),
                ms(trsm),
                ms(trailing),
                ms(zero),
                tiled,
                column,
                admitted,
            );
        }
    }
    ft_kernel_cpu::set_dgemm_bt_sub_tile_2d(false);
    println!(
        "\nREADING: a trusted isolation win is necessary, not sufficient. This arm stays DEFAULT \\
         OFF unless it also survives the paired live-PyTorch Cholesky lane with valid dual nulls."
    );
}
