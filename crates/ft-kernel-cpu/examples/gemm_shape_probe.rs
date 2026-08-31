//! Is the shared GEMM slow, or is it being handed a bad SHAPE? — `frankentorch-06csx`.
//!
//! WHY THIS EXISTS. Three beads converged on "the GEMM microkernel is the binding constraint":
//! the f32 conv dweight frame runs ~134 GFLOP/s (ledger 277), the f32 dinput route ~150 GFLOP/s
//! (276e), and `inv`'s f64 solve ~32 GFLOP/s against torch's implied ~78 (275). The natural
//! reading was "write a packed-panel Goto/BLIS kernel".
//!
//! THAT READING IS PROBABLY WRONG, AND CHECKING COSTS ONE PROBE. `gemm::sgemm` delegates to
//! `matrixmultiply`, which already IS a Goto-style packed-panel implementation with hand-written
//! micro-kernels. So "write a packed microkernel" is largely rediscovering a dependency we already
//! have, and the interesting question is not the kernel's peak rate but the SHAPE it is called at.
//!
//! THE SHAPE IT IS CALLED AT, and this is the specific thing under test. The streamed dweight
//! parallelises over the OUTPUT, and the output is tiny: `out_ch * patch_width` = 32 * 288 = 9216
//! elements, split across 16 threads, so each thread owns a 576-element tile and calls
//! `sgemm_tb_add_into(8, 256, 72)` about 640 times. Computing a 576-element tile requires
//! streaming all of K, so the arithmetic intensity is fixed at
//! `(8*72*2) / ((8 + 72) * 4)` = 3.6 flops/byte no matter how the tile is arranged — and the mb
//! sweep (277) already showed rearranging it does nothing.
//!
//! The alternative is SPLIT-K: give each thread a slice of K and the WHOLE 32x288 output, then
//! reduce. Same total work, but the per-thread call becomes `(32, 10240, 288)` — a well-shaped
//! GEMM at `(32*288*2) / ((32 + 288) * 4)` = 14.4 flops/byte, 4x the intensity.
//!
//! THE TWO ARMS ARE EXACTLY FLOP-MATCHED, which is what makes this a clean comparison rather than
//! two timings: 8*72*163840*2 and 32*288*10240*2 are both 188.7 MFLOP. Single-threaded, so the
//! only difference is the shape the microkernel sees.
//!
//! PREDICTION, RECORDED BEFORE THE RUN. SPLIT-K wins, and by a lot — 4x the arithmetic intensity
//! and a shape the kernel is actually designed for. If it does NOT, the microkernel is simply at
//! its rate for this problem, "the shape is wrong" is refuted, and bead 06csx should be closed
//! rather than turned into a kernel-writing project.
//!
//! WHAT A WIN HERE WOULD AND WOULD NOT LICENCE. Split-K changes the k-accumulation order, so it is
//! NOT bit-exact against the panel GEMM and would break
//! `conv2d_dweight_streamed_f32_matches_the_panel_gemm_bitwise`. This probe therefore decides
//! whether that parity question is worth ASKING; it does not answer it, and nothing here touches a
//! shipping path.
//!
//! ARM-INTERNAL: no incumbent, no ratio against torch, no drift gate. Everything to STDERR.

use std::time::Instant;

fn main() {
    // ARGV, not env: `rch exec` does not forward the caller's environment (ledger 273c).
    //   gemm_shape_probe [reps] [threads]
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let reps: usize = argv.first().and_then(|t| t.parse().ok()).unwrap_or(7);
    let threads: usize = argv.get(1).and_then(|t| t.parse().ok()).unwrap_or(1);
    if threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build_global()
            .expect("rayon pool width");
    }
    eprintln!(
        "PROV host={} nproc={} rayon={} reps={reps} loadavg={}",
        std::fs::read_to_string("/etc/hostname").unwrap_or_default().trim(),
        std::thread::available_parallelism().map_or(0, std::num::NonZero::get),
        rayon::current_num_threads(),
        std::fs::read_to_string("/proc/loadavg").unwrap_or_default().trim(),
    );

    let fill = |n: usize, seed: usize| -> Vec<f32> {
        (0..n).map(|i| ((i + seed) % 17) as f32 * 0.0625 - 0.5).collect()
    };

    // ---- ARM 1: the CEILING this box reaches on a shape the kernel likes. ------------------
    // A square GEMM is what matrixmultiply is tuned for, so this is the practical rate the
    // micro-kernel achieves here. Without it, "134 GFLOP/s" is a number with nothing to be slow
    // relative to.
    {
        let (m, k, n) = (1024usize, 1024usize, 1024usize);
        let a = fill(m * k, 1);
        let b = fill(k * n, 7);
        let mut c = vec![0.0f32; m * n];
        let mut best = f64::INFINITY;
        for rep in 0..reps {
            let start = Instant::now();
            ft_kernel_cpu::probe_sgemm(m, k, n, &a, &b, &mut c);
            let ms = start.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&c);
            if rep > 0 {
                best = best.min(ms);
            }
        }
        let gflop = 2.0 * (m * k * n) as f64 / 1e9;
        eprintln!(
            "GEMM CEILING   sgemm {m}x{k}x{n}          {best:8.3} ms   {:7.1} GFLOP/s",
            gflop / (best / 1e3)
        );
    }

    // ---- ARM 2: the shape the streamed dweight actually calls. ------------------------------
    // 640 sequential calls of (8, 256, 72), which is one thread's share of the real work.
    let tile_gflop = 2.0 * (8 * 72 * 163_840) as f64 / 1e9;
    {
        let (m, k, n) = (8usize, 256usize, 72usize);
        let calls = 163_840 / k;
        let a = fill(k * m, 3);
        let b = fill(k * n, 11);
        let mut c = vec![0.0f32; m * n];
        let mut best = f64::INFINITY;
        for rep in 0..reps {
            c.fill(0.0);
            let start = Instant::now();
            for _ in 0..calls {
                ft_kernel_cpu::probe_sgemm_tb_add_into(m, k, n, &a, &b, &mut c);
            }
            let ms = start.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&c);
            if rep > 0 {
                best = best.min(ms);
            }
        }
        eprintln!(
            "GEMM OUTPUT-PAR  {calls} x tb_add_into({m},{k},{n})  {best:8.3} ms   {:7.1} GFLOP/s   <- what dweight does now",
            tile_gflop / (best / 1e3)
        );
    }

    // ---- ARM 3: the shape SPLIT-K would call. -----------------------------------------------
    // One call of (32, 10240, 288): the same 188.7 MFLOP, the whole output tile, 4x the
    // arithmetic intensity.
    {
        let (m, k, n) = (32usize, 10_240usize, 288usize);
        let a = fill(k * m, 5);
        let b = fill(k * n, 13);
        let mut c = vec![0.0f32; m * n];
        let mut best = f64::INFINITY;
        for rep in 0..reps {
            c.fill(0.0);
            let start = Instant::now();
            ft_kernel_cpu::probe_sgemm_tb_add_into(m, k, n, &a, &b, &mut c);
            let ms = start.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&c);
            if rep > 0 {
                best = best.min(ms);
            }
        }
        eprintln!(
            "GEMM SPLIT-K     1 x tb_add_into({m},{k},{n})  {best:8.3} ms   {:7.1} GFLOP/s   <- same 188.7 MFLOP",
            tile_gflop / (best / 1e3)
        );
    }

    // ---- ARM 4: is it the CALL COUNT or the TILE SHAPE? -------------------------------------
    // Same tiny output tile as arm 2, but reached in ONE call with the full K instead of 640
    // calls of K=256. If arm 4 matches arm 2, per-call overhead is not the story and the tile
    // shape is; if arm 4 is much faster, the k-blocking that parity locks to SGEMM_KC is.
    {
        let (m, k, n) = (8usize, 163_840usize, 72usize);
        let a = fill(k * m, 17);
        let b = fill(k * n, 19);
        let mut c = vec![0.0f32; m * n];
        let mut best = f64::INFINITY;
        for rep in 0..reps {
            c.fill(0.0);
            let start = Instant::now();
            ft_kernel_cpu::probe_sgemm_tb_add_into(m, k, n, &a, &b, &mut c);
            let ms = start.elapsed().as_secs_f64() * 1e3;
            std::hint::black_box(&c);
            if rep > 0 {
                best = best.min(ms);
            }
        }
        eprintln!(
            "GEMM ONE-CALL    1 x tb_add_into({m},{k},{n})  {best:8.3} ms   {:7.1} GFLOP/s   <- arm 2's shape, one call",
            tile_gflop / (best / 1e3)
        );
    }
}
