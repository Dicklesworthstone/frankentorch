//! `frankentorch-mdsmm` — does the board's conv2d lane actually REACH the all-ones fast path?
//!
//! # The question, and why it has to be measured rather than read
//!
//! Every lane on `gauntlet_lane_sweep_h2h` ends its timed region in `.sum().backward()`, which
//! makes `dout` all ones. conv2d has all-ones fast paths gated on SHAPE — `Height1` and
//! `ThreeByThreeStride1` — so whether a given lane is scored on the fast path or on the route a
//! real objective reaches depends entirely on that lane's fixture.
//!
//! The board's conv2d fixture is batch 8, 32 channels in and out, 32x32, k3x3, stride (1,1),
//! padding (1,1) — so ph = 34 and oh = 32, and the `ThreeByThreeStride1` predicate
//! (`ph > 1 && oh > 1 && kh == 3 && kw == 3 && sh == 1 && sw == 1`) is satisfied on paper.
//!
//! **On paper is not good enough.** AGENTS.md is explicit that grepping to a match and assuming it
//! is the live one is how you fix dead code, and this repo has three recorded cases of confident
//! source reading being wrong about which path executes. So this censuses the route by OBSERVING
//! it, using an instrument that already exists.
//!
//! # The observable
//!
//! `conv2d_backward_gemm_counts()` counts the dweight and dpanel GEMMs. The all-ones adjoints
//! replace both with a column-sum and a single one-row GEMM, so **on the fast route there is no
//! GEMM to count**: a reading of `(0, 0)` is the fast path, and non-zero counts are the generic
//! route. Those counters became trustworthy at ledger 293d/293f — before that a concurrent test
//! could contribute to them, and before vcxf7 the streamed dweight could read off the end of a
//! buffer entirely.
//!
//! Running the SAME shape twice, once with an all-ones `dout` and once with a non-uniform one, is
//! the control: if the counts differ, the two losses demonstrably execute different code, which is
//! the whole claim the bead rests on. If they do not differ, this lane has no blind spot and needs
//! no twin — and item 109's premise would not apply to it.
//!
//! COUNTS, NOT TIMINGS. Valid under any host load, no measurement window required.
//!
//!   cargo run --release -p frankentorch-kernel-cpu --example conv2d_ones_route_census

/// The board's conv2d lane fixture, read from `gauntlet_lane_sweep_h2h.rs`:
/// `C2_N=8, C2_CI=32, C2_CO=32, C2_H=32, C2_W=32, C2_K=3`, `functional_conv2d(.., (1,1), (1,1))`.
const BATCH: usize = 8;
const IN_CH: usize = 32;
const OUT_CH: usize = 32;
const K: usize = 3;
const PAD: usize = 1;
const H: usize = 32;

fn main() {
    let ph = H + 2 * PAD;
    let pw = H + 2 * PAD;
    let oh = (ph - K) / 1 + 1;
    let ow = (pw - K) / 1 + 1;
    println!(
        "CONV2D BOARD LANE FIXTURE: batch={BATCH} in_ch={IN_CH} out_ch={OUT_CH} {H}x{H} k{K}x{K} \
         stride 1 pad {PAD}  ->  padded {ph}x{pw}, out {oh}x{ow}"
    );
    println!(
        "PREDICATE ThreeByThreeStride1 = ph>1 && oh>1 && kh==3 && kw==3 && sh==1 && sw==1 -> {}",
        ph > 1 && oh > 1 && K == 3
    );
    println!(
        "OBSERVABLE: conv2d_backward_gemm_counts(). The all-ones adjoints replace BOTH GEMMs with \
         a column-sum plus one one-row GEMM, so (0, 0) means the fast path ran and non-zero means \
         the generic route did. Counts, not timings — valid under any load.\n"
    );

    let padded: Vec<f64> = (0..BATCH * IN_CH * ph * pw)
        .map(|i| ((i % 37) as f64) * 0.013 - 0.21)
        .collect();
    let weight: Vec<f64> = (0..OUT_CH * IN_CH * K * K)
        .map(|i| ((i % 11) as f64) * 0.0625 - 0.3125)
        .collect();
    let padded32: Vec<f32> = padded.iter().map(|&v| v as f32).collect();
    let weight32: Vec<f32> = weight.iter().map(|&v| v as f32).collect();

    let n_dout = BATCH * OUT_CH * oh * ow;
    // `.sum().backward()` produces exactly this.
    let ones: Vec<f64> = vec![1.0; n_dout];
    // What any real objective produces.
    let varied: Vec<f64> = (0..n_dout).map(|i| ((i % 23) as f64) * 0.019 - 0.19).collect();
    let ones32: Vec<f32> = vec![1.0; n_dout];
    let varied32: Vec<f32> = varied.iter().map(|&v| v as f32).collect();

    println!("  {:>10} {:>8} {:>22} {:>6} {:>19}", "dtype", "loss", "(dweight, dpanel) GEMMs", "ones", "route");
    for (label, dout) in [("sum()", &ones), ("non-uniform", &varied)] {
        ft_kernel_cpu::reset_conv2d_backward_gemm_counts();
        let _ = ft_kernel_cpu::conv2d_ones_path_census_take();
        let out = ft_kernel_cpu::conv2d_backward_f64(
            dout, &padded, &weight, BATCH, IN_CH, ph, pw, K, K, oh, ow, 1, 1, OUT_CH, false,
        );
        let counts = ft_kernel_cpu::conv2d_backward_gemm_counts();
        let ones_hits = ft_kernel_cpu::conv2d_ones_path_census_take();
        std::hint::black_box(&out);
        let route = if ones_hits > 0 { "ALL-ONES FAST PATH" } else { "generic" };
        println!("  {:>10} {label:>8} {:>22} {ones_hits:>6} {route:>19}", "f64", format!("{counts:?}"));
    }
    for (label, dout) in [("sum()", &ones32), ("non-uniform", &varied32)] {
        ft_kernel_cpu::reset_conv2d_backward_gemm_counts();
        let _ = ft_kernel_cpu::conv2d_ones_path_census_take();
        let out = ft_kernel_cpu::conv2d_backward_f32(
            dout, &padded32, &weight32, BATCH, IN_CH, ph, pw, K, K, oh, ow, 1, 1, OUT_CH, false,
        );
        let counts = ft_kernel_cpu::conv2d_backward_gemm_counts();
        let ones_hits = ft_kernel_cpu::conv2d_ones_path_census_take();
        std::hint::black_box(&out);
        let route = if ones_hits > 0 { "ALL-ONES FAST PATH" } else { "generic" };
        println!("  {:>10} {label:>8} {:>22} {ones_hits:>6} {route:>19}", "f32", format!("{counts:?}"));
    }

    println!(
        "\nREADING: if the two loss shapes give DIFFERENT counts, they demonstrably execute \
         different code and every lane on this fixture without a non-uniform twin is scoring a \
         branch training never reaches. If they give the SAME counts, this fixture has no blind \
         spot and needs no twin — which would be worth knowing before writing four of them."
    );
}
