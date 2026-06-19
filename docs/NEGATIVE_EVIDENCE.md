# FrankenTorch Negative-Evidence Ledger

This ledger records optimization attempts that failed, regressed, or did not
clear the benchmark bar. Do not retry a rejected lever unless the retry condition
is explicitly satisfied.

## 2026-06-19 - frankentorch-kgs4.122 - AvgPool1d unit-dy fill

- Lever: special-case f64 `avg_pool1d_backward_f64` for kernel `2`, stride `2`,
  exact full coverage, and all-ones `dout`, returning a constant `0.5` gradient
  fill instead of the generic accumulation loop.
- Workload: `gauntlet_avg_pool1d_grad`, deterministic f64
  `[N,C,L]=[8,64,8192]`, kernel `2`, stride `2`, forward
  `functional_avg_pool1d`, scalar `sum`, backward.
- Reference: local PyTorch `2.12.1+cpu` in
  `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`, 32 compute
  threads and 32 interop threads on `thinkstation1`.
- Candidate result with the fast path: FrankenTorch median `204.02 ms`;
  PyTorch median `7.4798 ms`; ratio vs PyTorch `27.28x` slower.
- Current-minus-fast-path baseline: FrankenTorch median `179.91 ms`; PyTorch
  median `7.0626 ms`; ratio vs PyTorch `25.47x` slower.
- Final reverted-source result: FrankenTorch median `184.99 ms`; PyTorch median
  `7.1539 ms`; ratio vs PyTorch `25.86x` slower.
- Final reverted-source rerun: FrankenTorch median `181.94 ms`; PyTorch median
  `7.3011 ms`; ratio vs PyTorch `24.92x` slower.
- Candidate vs fast-path-disabled baseline: `1.134x` slower by median.
- Verdict: rejected and reverted. The standalone all-ones `dout` constant-fill
  branch regressed the realistic full training-style workload and must not be
  retried as a tiny avg_pool1d backward-only lever.
- Retry condition: Retry only if a profiler attributes a clearly-above-noise
  share to `avg_pool1d_backward_f64` fill/scatter work on the full
  `gauntlet_avg_pool1d_grad` training workload after forward, session/tape
  setup, allocation churn, and tensor materialization overhead are separated.
  Otherwise target end-to-end pooling overhead instead of another
  constant-gradient branch.
- Evidence:
  - `artifacts/perf/frankentorch-kgs4.122/gauntlet_20260619T0358Z/current_criterion_avg_pool1d.log`
  - `artifacts/perf/frankentorch-kgs4.122/gauntlet_20260619T0358Z/current_env.txt`
  - `artifacts/perf/frankentorch-kgs4.122/gauntlet_20260619T0358Z/baseline_minus_fastpath_criterion_avg_pool1d.log`
  - `artifacts/perf/frankentorch-kgs4.122/gauntlet_20260619T0358Z/baseline_minus_fastpath_env.txt`
  - `artifacts/perf/frankentorch-kgs4.122/gauntlet_20260619T0358Z/final_reverted_criterion_avg_pool1d.log`
  - `artifacts/perf/frankentorch-kgs4.122/gauntlet_20260619T0358Z/final_reverted_env.txt`
  - `artifacts/perf/frankentorch-kgs4.122/gauntlet_20260619T0358Z/rerun_current_criterion_avg_pool1d.log`

## 2026-06-19 - frankentorch-kgs4.117 - MaxPool3d saved-index sidecar

- Lever: save compact f64 max-pool3d first-argmax offsets during forward and
  scatter backward gradients from that sidecar instead of saving the full input
  and rescanning each 2x2x2 window during backward.
- Workload: `gauntlet_max_pool3d_grad`, deterministic f64
  `[N,C,D,H,W]=[2,32,16,32,32]`, kernel `2x2x2`, stride `2x2x2`,
  forward max_pool3d, scalar `sum`, backward.
- Reference: local PyTorch `2.12.1+cpu` in
  `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`, 32 compute
  threads and 32 interop threads on `thinkstation1`.
- Parent-before-sidecar result at `c79d3a23`: FrankenTorch median `20.585 ms`;
  PyTorch median `2.1381 ms`; ratio vs PyTorch `9.63x` slower.
- Current post-lint result at `7cbaf731` plus clippy-only lint fixes:
  FrankenTorch median `15.794 ms`; PyTorch median `1.6228 ms`; ratio vs
  PyTorch `9.73x` slower. This is a `1.30x` internal FrankenTorch speedup vs
  the parent-before-sidecar row, but not PyTorch dominance.
- Supplemental remote row: rch `hz2` built the bench and measured current
  FrankenTorch at `28.124 ms`, then failed the PyTorch arm because the worker
  did not have `torch` installed. Treat this as build/FT-only evidence, not as
  a ratio-vs-PyTorch result.
- Verdict: keep as a measured internal win; classify as a PyTorch-loss row for
  release readiness. No source revert.
- Retry condition: do not retry max_pool3d sidecar-only or rescan-only variants
  unless a fresh profile proves saved-context memory or backward window rescans
  still dominate after session setup, allocation churn, and tensor materializing
  costs are separated. The next max_pool3d gap-closing pass should target the
  end-to-end PyTorch gap, not another standalone sidecar shape tweak.
- Evidence:
  - `artifacts/perf/frankentorch-kgs4.117/gauntlet_20260619T0320Z/parent_local_warm_criterion.txt`
  - `artifacts/perf/frankentorch-kgs4.117/gauntlet_20260619T0320Z/current_local_warm_postlint_criterion.txt`
  - `artifacts/perf/frankentorch-kgs4.117/gauntlet_20260619T0320Z/current_criterion.txt`
  - `artifacts/perf/frankentorch-kgs4.117/gauntlet_20260619T0320Z/ft_kernel_cpu_max_pool3d_sidecar_test_postlint.log`
  - `artifacts/perf/frankentorch-kgs4.117/gauntlet_20260619T0320Z/ft_api_max_pool3d_grad_test_postlint.log`
  - `artifacts/perf/frankentorch-kgs4.117/gauntlet_20260619T0320Z/ft_api_bench_clippy_postlint.log`

## 2026-06-19 - frankentorch-kgs4.128 - MaxPool3d end-to-end profile rejects

- Levers:
  - Borrowed-input custom autograd route for f64 `functional_max_pool3d` grad
    fast path, replacing the owned-input `tensor_apply_function` materialization.
  - Exact all-ones `dout` backward scatter branch from saved max-pool3d argmax
    offsets, tested as both rayon plane-parallel and sequential plane-local
    variants.
- Workload: `gauntlet_max_pool3d_grad`, deterministic f64
  `[N,C,D,H,W]=[2,32,16,32,32]`, kernel `2x2x2`, stride `2x2x2`,
  forward max_pool3d, scalar `sum`, backward.
- Reference: local PyTorch `2.12.1+cpu` in
  `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`, 32 compute
  threads and 32 interop threads on `thinkstation1`.
- Clean baseline: FrankenTorch median `15.303 ms`; PyTorch median `1.6325 ms`;
  ratio vs PyTorch `9.38x` slower.
- Stage baseline: setup tensor `215.47 us`; FrankenTorch forward-only
  `4.1256 ms`; sum-only `1.3121 ms`; backward-only `43.433 ms` with severe
  outliers; raw kernel forward+indices `727.15 us`; raw kernel backward
  from indices `9.0069 ms` with severe outliers. Treat the stage probe as
  routing evidence, not ratio proof.
- Borrowed-input candidate: headline FrankenTorch median `22.764 ms`; PyTorch
  median `1.6633 ms`; ratio vs PyTorch `13.69x` slower. The isolated
  forward-only stage improved from `4.1256 ms` to `1.8935 ms`, but the full
  workload regressed `1.49x` vs the clean baseline. Rejected and reverted.
- Rayon all-ones `dout` candidate: headline FrankenTorch median `16.160 ms`;
  PyTorch median `1.6543 ms`; ratio vs PyTorch `9.77x` slower. This was
  `1.06x` slower than the clean baseline. Rejected and reverted.
- Sequential all-ones `dout` candidate: headline FrankenTorch median
  `22.465 ms`; the paired PyTorch row had severe high outliers, so using the
  clean PyTorch baseline gives a routing ratio of `13.76x` slower. Rejected and
  reverted.
- Final reverted-source sanity row: FrankenTorch median `16.586 ms`; paired
  PyTorch row had severe high outliers and is not primary ratio evidence.
- Verdict: no product source kept. The durable result is negative evidence plus
  a stage-probe benchmark harness for future max_pool3d gap work.
- Retry condition: do not retry borrowed-input-only max_pool3d routes or
  standalone unit-`dout` scatter branches. Revisit only with a fusion that
  removes the sum-generated gradient buffer/tape edge end-to-end, an allocator
  or arena change proven on the whole training row, or a fundamentally different
  kernel/layout plan with fresh same-workload ratio evidence.
- Evidence:
  - `artifacts/perf/frankentorch-kgs4.128/gauntlet_20260619T0521Z/env.txt`
  - `artifacts/perf/frankentorch-kgs4.128/gauntlet_20260619T0521Z/baseline_criterion_max_pool3d.log`
  - `artifacts/perf/frankentorch-kgs4.128/gauntlet_20260619T0521Z/baseline_stage_max_pool3d.log`
  - `artifacts/perf/frankentorch-kgs4.128/gauntlet_20260619T0521Z/candidate_criterion_max_pool3d.log`
  - `artifacts/perf/frankentorch-kgs4.128/gauntlet_20260619T0521Z/unit_dout_candidate_criterion_max_pool3d.log`
  - `artifacts/perf/frankentorch-kgs4.128/gauntlet_20260619T0521Z/unit_dout_sequential_candidate_criterion_max_pool3d.log`
  - `artifacts/perf/frankentorch-kgs4.128/gauntlet_20260619T0521Z/final_reverted_criterion_max_pool3d.log`
  - `artifacts/perf/frankentorch-kgs4.128/gauntlet_20260619T0521Z/summary.md`

## 2026-06-19 - frankentorch-kgs4.121 - Linear all-ones dy kernel move

- Lever: detect exact all-ones `dy` from `tensor_linear(...).sum().backward()`
  and collapse the f64 linear backward into row-sum/copy work instead of the
  generic two-GEMM backward.
- Workload: `gauntlet_linear_train_hidden_2048`, deterministic f64
  `[batch,in]=[32,512]`, `[hidden,in]=[2048,512]`, f64 bias, linear forward,
  scalar `sum`, backward.
- Reference: local PyTorch `2.12.1+cpu` in
  `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`, 32 compute
  threads and 32 interop threads on `thinkstation1`.
- Parent-before-lever result at `4d1198f9`: FrankenTorch median `29.606 ms`;
  PyTorch median `9.8492 ms`; ratio vs PyTorch `3.01x` slower.
- API-local candidate result at `b5bca44e`: FrankenTorch median `21.494 ms`;
  PyTorch median `8.6461 ms`; ratio vs PyTorch `2.49x` slower. This is a
  `1.38x` internal FrankenTorch speedup vs the parent-before-lever row.
- Kernel-move candidate result at `81032a4d`: FrankenTorch median `26.459 ms`;
  PyTorch median `9.7925 ms`; ratio vs PyTorch `2.70x` slower. This regressed
  the API-local row by `1.23x`.
- Final restored-path result after reverting the kernel move: FrankenTorch
  median `22.775 ms`; PyTorch median `9.2821 ms`; ratio vs PyTorch `2.45x`
  slower. This is a `1.30x` internal speedup vs parent-before-lever, but not
  PyTorch dominance.
- Verdict: keep the API-local all-ones `dy` helper as a measured internal win;
  reject and revert the kernel-level relocation. `frankentorch-kgs4.121` is
  measured, not pending.
- Retry condition: do not retry tiny kernel-level all-ones GEMM replacement
  variants for this workload. Revisit only if a fresh profile shows linear
  backward row-fill/reduction dominates after tape, allocation, and forward
  setup are separated, or if a broader linear-training lever closes the
  PyTorch gap end-to-end.
- Evidence:
  - `artifacts/perf/frankentorch-kgs4.121/gauntlet_20260619T0325Z/prelever_criterion_linear.log`
  - `artifacts/perf/frankentorch-kgs4.121/gauntlet_20260619T0325Z/baseline_criterion_linear.log`
  - `artifacts/perf/frankentorch-kgs4.121/gauntlet_20260619T0325Z/current_criterion_linear.log`
  - `artifacts/perf/frankentorch-kgs4.121/gauntlet_20260619T0325Z/final_criterion_linear.log`
  - `artifacts/perf/frankentorch-kgs4.121/gauntlet_20260619T0325Z/final_env.txt`

## 2026-06-19 - frankentorch-kgs4.124 - SmoothL1 direct reduced grad

- Lever: route same-shape f64 `tensor_smooth_l1_loss(..., reduction="mean")`
  through a scalar reduced autograd op instead of materializing the full
  per-element SmoothL1 output and uniform backward `dloss`.
- Workload: `smooth_l1/grad_8m`, 8,388,608 f64 elements, mean reduction,
  forward loss plus backward.
- Reference: local PyTorch `2.12.1+cu130` CPU path, 32 compute threads.
- Decisive internal A/B: same-worker `hz2` Criterion pre-lever median
  `963.16 ms`; current median `757.63 ms`; FrankenTorch internal speedup
  `1.27x`.
- PyTorch head-to-head: local current FrankenTorch median `742.95 ms`;
  local PyTorch median `373.61 ms`; FrankenTorch/PyTorch time ratio
  `1.99x` slower.
- Supplemental drift row: unpinned current FrankenTorch on `ovh-a` measured
  `595.82 ms`; this row is routing evidence only because the pre-lever row
  ran on `hz2`.
- Verdict: kept as a measured FrankenTorch internal win, but not counted as
  PyTorch dominance. No source revert. `frankentorch-kgs4.124` is closed.
- Retry condition: do not retry scalar reduced-loss wrapper variants. The
  follow-up `frankentorch-kgs4.128` must attack deeper tape, allocation,
  loss-kernel, SIMD, or cache-layout cost until this row beats PyTorch.
- Evidence:
  - `artifacts/perf/frankentorch-kgs4-smooth-l1-reduced-grad/gauntlet_20260619/prelever_81032a4d_criterion_smooth_l1_grad_8m.log`
  - `artifacts/perf/frankentorch-kgs4-smooth-l1-reduced-grad/gauntlet_20260619/current_hz2_criterion_smooth_l1_grad_8m.log`
  - `artifacts/perf/frankentorch-kgs4-smooth-l1-reduced-grad/gauntlet_20260619/current_local_criterion_smooth_l1_grad_8m.log`
  - `artifacts/perf/frankentorch-kgs4-smooth-l1-reduced-grad/gauntlet_20260619/torch_smooth_l1_grad_8m_local.json`

## 2026-06-19 - frankentorch-kgs4.126 - max_pool1d unit-dout scatter

- Lever: special-case `functional_max_pool1d` f64 backward when `dout` is exact
  all-ones, scattering `1.0` directly from saved argmax offsets.
- Workload: `gauntlet_max_pool1d_grad`, `[N,C,L]=[8,64,8192]`, kernel `2`,
  stride `2`, f64 leaf, forward max_pool1d, `sum`, backward.
- Reference: PyTorch `2.12.1+cpu` in
  `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`.
- Host: `thinkstation1`, `nproc=64`, PyTorch compute threads `32`, interop
  threads `32`.
- Candidate result at `ae4ace3b`: FrankenTorch median `184.41 ms`; PyTorch
  median `14.984 ms`; ratio vs PyTorch `12.31x` slower.
- Parent-before-lever result at `eda26661`: FrankenTorch median `178.47 ms`;
  PyTorch median `16.199 ms`; ratio vs PyTorch `11.02x` slower.
- Candidate vs parent: `1.033x` slower by median; Criterion reported no
  statistically significant improvement (`p=0.12`, no performance change).
- Verdict: rejected and reverted. The exact-unit `dout` branch does not improve
  the realistic full training-style workload and should not be retried as a
  standalone max_pool1d backward lever.
- Retry condition: only revisit if profiling proves max_pool1d backward scatter
  itself is a dominant self-time frame after forward/session/allocation overhead
  is removed, or if a broader allocation-elision/autograd-tape lever changes the
  workload cost model.
- Evidence:
  - `artifacts/perf/frankentorch-kgs4.126/gauntlet_20260619T0113Z/criterion.txt`
  - `artifacts/perf/frankentorch-kgs4.126/gauntlet_20260619T0113Z/baseline_criterion.txt`
  - `artifacts/perf/frankentorch-kgs4.126/gauntlet_20260619T0113Z/env.txt`
  - `artifacts/perf/frankentorch-kgs4.126/gauntlet_20260619T0113Z/baseline_env.txt`

## Historical SmoothL1/loss guardrails

- Rejected: f32 SmoothL1 no-grad fused path in
  `artifacts/perf/frankentorch-cs2d/rejected_f32_smooth_l1_fast_path.md`.
  Do not retry without a fresh dtype audit and same-worker A/B.
- Kept: f64 SmoothL1 no-grad pairwise reducer in
  `artifacts/perf/frankentorch-ruby-smoothl1-f64-reduction/report.md`,
  baseline `136.80 ms` to `97.302 ms`. Do not rework the no-grad reducer
  family for the grad bead.
- Rejected: direct reduced Gaussian NLL grad in `frankentorch-fdn1v`,
  `829.27 ms` to `1.0274 s`. Do not generalize the SmoothL1 reduced-grad
  lever to Gaussian NLL without new profile proof.
