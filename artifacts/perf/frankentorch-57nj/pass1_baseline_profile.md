# frankentorch-57nj pass 1 baseline/profile

Date: 2026-06-07T03:58Z-04:01Z UTC.
Skill: extreme-software-optimization.
Scope: baseline/profile only; no source edits.

## Tracker state

`br show frankentorch-57nj --json` currently reports this bead as `closed`,
assigned to `RubyLotus`, with the rejection reason that per-batch backward
skip-transpose reassociates the `K=flat` dweight reduction and is not
bit-exact.

`br ready --json` returned `[]` during this pass.

## Commands

- `br show frankentorch-57nj --json`
- `br ready --json`
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- conv2d/grad_hw/64 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- `CARGO_TARGET_DIR=artifacts/perf/frankentorch-57nj/local-target perf record -F 99 --call-graph dwarf -o artifacts/perf/frankentorch-57nj/perf_conv2d_grad_hw64_bench.data -- ./.rch-target-ts1-job-29872254800115103-1780804708624525236-0/release/deps/ops_bench-5797319cbc2b6290 --bench conv2d/grad_hw/64 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- `perf report --stdio -g none --no-children --percent-limit 0.1 -i artifacts/perf/frankentorch-57nj/perf_conv2d_grad_hw64_bench.data`

## Baseline

Remote worker: `ts1` (`ubuntu@192.168.1.107`).

Criterion row:

```text
conv2d/grad_hw/64       time:   [200.74 ms 208.75 ms 217.24 ms]
                        thrpt:  [19.308 Melem/s 20.093 Melem/s 20.895 Melem/s]
```

Criterion JSON pulled back by `rch`:
`.rch-target-ts1-job-29872254800115103-1780804708624525236-0/criterion/conv2d/grad_hw/64/new/estimates.json`

Mean point estimate: `208.74932016666666 ms`.
Median point estimate: `206.5983326666667 ms`.

## Golden SHA

`sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`
passed for all locally present entries.

## dout_t materialization evidence

Current source path: `crates/ft-kernel-cpu/src/lib.rs`.

For the `conv2d/grad_hw/64` row:

```text
batch=4, in_ch=64, out_ch=64, kh=3, kw=3, oh=64, ow=64
patch_count=4096
flat=16384
patch_width=576
dout_flat = 1,048,576 f64 = 8.00 MiB
dout_t    = 1,048,576 f64 = 8.00 MiB
panel     = 9,437,184 f64 = 72.00 MiB
dpanel    = 9,437,184 f64 = 72.00 MiB
dweight GEMM multiply-adds = 603,979,776
dpanel GEMM multiply-adds  = 603,979,776
two backward GEMMs total   = 1,207,959,552 multiply-adds
```

The `dout_t` section is:

- gather `dout_flat [flat, out_ch]` into `dout_t [out_ch, flat]`
- pure copy, no arithmetic
- exact write volume is 8.00 MiB plus strided reads from `dout_flat`
- feeds `gemm::dgemm(out_ch, flat, patch_width, &dout_t, &panel, &mut dweight)`

Local perf profile, benchmark mode:
`artifacts/perf/frankentorch-57nj/perf_conv2d_grad_hw64_bench.data`.

Selected flat samples from `perf report`:

```text
12.32%  libm.so.6                   __ieee754_exp_fma
 5.67%  ops_bench                   <ft_autograd::TensorTape>::backward_with_options
 4.81%  libc.so.6                   __memmove_avx_unaligned_erms
 4.38%  ops_bench                   matrixmultiply::dgemm_kernel::kernel_target_fma
 2.74%  ops_bench                   <ft_autograd::TensorTape>::pad
 1.17%  ops_bench                   matrixmultiply::packing::pack_avx2::<matrixmultiply::kernel::U8, f64>
 0.95%  ops_bench                   conv2d_im2col_f64::{closure#0}
 0.92%  ops_bench                   conv2d_col2im_f64::{closure#0}
 0.47%  ops_bench                   rayon helper for conv2d_backward_f64::{closure#1}
 0.24%  ops_bench                   conv2d_backward_f64::{closure#1}
```

Interpretation: the existing `dout_t` materialization is visible but small in
this whole-benchmark profile. The full row is currently dominated by RNG
generation/libm, autograd/tape work, memory movement, GEMM kernel/packing, and
im2col/col2im. A future transpose-free dweight primitive must preserve the
single `K=flat` accumulation order to keep bit-exactness.

## Files created

- `artifacts/perf/frankentorch-57nj/perf_conv2d_grad_hw64.data`
  - first local perf capture, direct binary invocation/test-mode banner.
- `artifacts/perf/frankentorch-57nj/perf_conv2d_grad_hw64_bench.data`
  - primary local perf capture, Criterion benchmark mode.
- `artifacts/perf/frankentorch-57nj/local-target/criterion/**`
  - small Criterion report files from local benchmark-mode sanity/profile runs.
- `artifacts/perf/frankentorch-57nj/pass1_baseline_profile.md`
  - this report.

No source files were intentionally edited during this pass.
