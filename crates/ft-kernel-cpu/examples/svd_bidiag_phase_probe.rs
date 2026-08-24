//! Same-worker A/B for the blocked bidiagonalization
//! (frankentorch-svd-blocked-bidiag-r7jdo).
//!
//! The reduction is chosen deep inside the kernel and latched once per process,
//! so the two arms cannot share a process. Instead the parent re-executes ITSELF
//! once per arm, interleaved, twice over: both arms therefore come from one ELF
//! on one worker under the same load, which is what the perf ledger requires of
//! an incumbent arm. Each child also runs an A/A null gate — two independent
//! timings of the identical arm — whose spread bounds how much of any A/B
//! difference is worker noise rather than the lever.
//!
//! ```text
//! cargo run --release -p ft-kernel-cpu --example svd_bidiag_phase_probe
//! ```

use ft_core::{DType, Device, TensorMeta};
use std::time::Instant;

fn matrix(m: usize, n: usize) -> Vec<f64> {
    let mut a = vec![0.0f64; m * n];
    let mut state = 0x2545_f491_4f6c_dd1du64;
    for value in &mut a {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *value = ((state >> 11) as f64) / ((1u64 << 53) as f64) - 0.5;
    }
    a
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(f64::total_cmp);
    xs[xs.len() / 2]
}

fn time<F: FnMut()>(reps: usize, mut f: F) -> f64 {
    // One untimed warm-up: the first call faults in the output buffers.
    f();
    let mut samples = Vec::with_capacity(reps);
    for _ in 0..reps {
        let start = Instant::now();
        f();
        samples.push(start.elapsed().as_secs_f64() * 1e3);
    }
    median(samples)
}

fn measure() {
    let arm = if std::env::var_os("FT_SVD_FORCE_NR").is_some() {
        "INCUMBENT(nr)"
    } else {
        "BLOCKED(dgebrd)"
    };
    let threads = rayon::current_num_threads();

    for &n in &[128usize, 256, 512] {
        let m = n;
        let a = matrix(m, n);
        let meta = TensorMeta::from_shape(vec![m, n], DType::F64, Device::Cpu);
        let reps = if n >= 512 { 3 } else { 5 };

        let aa1 = time(reps, || {
            std::hint::black_box(ft_kernel_cpu::svd_contiguous_f64(&a, &meta, false).unwrap());
        });
        let aa2 = time(reps, || {
            std::hint::black_box(ft_kernel_cpu::svd_contiguous_f64(&a, &meta, false).unwrap());
        });
        let full = time(reps, || {
            std::hint::black_box(ft_kernel_cpu::svd_contiguous_f64(&a, &meta, true).unwrap());
        });
        let vals = time(reps, || {
            std::hint::black_box(ft_kernel_cpu::svdvals_contiguous_f64(&a, &meta).unwrap());
        });

        let skew = (aa1 - aa2).abs() / aa1.max(aa2) * 100.0;
        println!(
            "{arm:<16} t={threads:<3} N={n:<4} reduced {aa1:8.2} ms (A/A {aa2:8.2}, skew \
             {skew:4.1}%)  full {full:8.2} ms  svdvals {vals:8.2} ms"
        );

        // FUSED-TRAILING A/B — `frankentorch-4zjaa`, NEGATIVE_EVIDENCE item 247b.
        //
        // ARM-INTERNAL, AND THEREFORE MAINTENANCE, NOT A WIN. There is no incumbent in this
        // process and no ratio against PyTorch; a self-speedup does not certify anything. What
        // this is for is SIZING: item 247b says the lever's "payoff is a memory-traffic argument
        // that only a measurement can settle", and knowing whether it is worth 2% or 20% decides
        // whether it deserves one of this host's rare quiet windows for a paired vs-incumbent
        // row. Cheap to take, and it needs only `ft-kernel-cpu`, which matters because every h2h
        // harness lives in `ft-api`.
        //
        // ONE PROCESS, palindrome ON/OFF/OFF/ON (item 51): the toggle is an `AtomicBool`, not the
        // `OnceLock` that forces the NR/blocked arms above to re-exec, so host drift lands
        // symmetrically on both arms instead of between two child processes. Min of the two
        // placements per arm, per this campaign's estimator convention on a shared host.
        //
        // The A/A skew printed above is the noise floor this ratio has to clear: two timings of
        // the IDENTICAL arm, so any fused/2pass difference smaller than that skew is not a
        // finding.
        let one = |fused: bool| -> f64 {
            let previous = ft_kernel_cpu::bidiag_fused_trailing_set(fused);
            let ms = time(reps, || {
                std::hint::black_box(ft_kernel_cpu::svdvals_contiguous_f64(&a, &meta).unwrap());
            });
            ft_kernel_cpu::bidiag_fused_trailing_set(previous);
            ms
        };
        let f1 = one(true);
        let p1 = one(false);
        let p2 = one(false);
        let f2 = one(true);
        let fused_ms = f1.min(f2);
        let twopass_ms = p1.min(p2);
        println!(
            "{arm:<16} t={threads:<3} N={n:<4} TRAILING svdvals: fused {fused_ms:8.2} ms  \
             2pass {twopass_ms:8.2} ms  2pass/fused {:.3}x  (>1 = the fusion helps; \
             arm-internal, NOT a vs-incumbent claim; A/A skew {skew:4.1}% is the floor)",
            twopass_ms / fused_ms
        );
    }
}

fn main() {
    if std::env::var_os("FT_SVD_AB_CHILD").is_some() {
        measure();
        return;
    }
    let exe = std::env::current_exe().expect("current_exe");
    for round in 1..=2 {
        println!("--- round {round} ---");
        for force_nr in [true, false] {
            let mut cmd = std::process::Command::new(&exe);
            cmd.env("FT_SVD_AB_CHILD", "1");
            if force_nr {
                cmd.env("FT_SVD_FORCE_NR", "1");
            }
            let status = cmd.status().expect("spawn arm");
            assert!(status.success(), "arm exited with {status}");
        }
    }
}
