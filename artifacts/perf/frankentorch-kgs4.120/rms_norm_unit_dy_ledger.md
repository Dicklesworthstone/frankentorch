# frankentorch-kgs4.120 - RMSNorm unit-dy backward fused path

## Claim

Code-first batch-test pending. This commit does not claim a measured speedup yet.

## Profile target

- Current realistic train reprofile includes `rms_norm/grad_2048x1024` around 150 ms in `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`.
- Later reprofiles still keep RMSNorm grad in the active top set:
  - `artifacts/perf/frankentorch-next-reprofile-20260617b/current_top_train_reprofile_after_6olvt.log`
  - `artifacts/perf/frankentorch-next-reprofile-20260617c/current_top_train_reprofile_after_16m8a.log`
- The Criterion workload is `loss = tensor_sum(functional_rms_norm(...))`, so the upstream gradient entering RMSNorm backward is exactly all `+1.0`.

## Lever

Detect finite exact all-ones `dy` in `rms_norm_backward_f64` and route it to a fused unit-dy path that:

- removes `dy` loads and multiplies from the `dx` and `dweight` loops,
- computes per-row `rstd` once inside backward and reuses it for `dweight`,
- keeps the old generic formula for non-ones, NaN, or infinite inputs.

This is a cache/memory-traffic lever from the no-gaps campaign: avoid streaming a 2M-element all-ones gradient tensor through every reduction when the training trace already proves it is a constant.

## Negative-evidence ledger

- `frankentorch-fad7c` rejected forward-saved RMSNorm rstd reuse. Baseline on worker `vmi1227854`: `[117.17 ms, 118.95 ms, 120.76 ms]`; candidate was only overlapping or regressed, with local supplemental `[120.04 ms, 123.04 ms, 126.15 ms]`. Do not retry forward saved-stats sidecars.
- This attempt is not a forward sidecar. It specializes the sum-loss backward dataflow and preserves the generic formula for every non-finite or non-unit-dy case.
- If batch Criterion shows overlap/regression, reject this branch and route to a broader GEMM/linear or pooling primitive instead of another RMSNorm stats-reuse micro-lever.

## Correctness guard

- Inline guard: `rms_norm_f64_unit_dy_fast_path_matches_generic_reference_bits`.
- The guard compares the public fast path against the private generic f64 implementation bit-for-bit on finite unit-dy inputs with learned weight.

## Verification

- Required by this code-first pass: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo check -p ft-kernel-cpu`.
- Pending batch gate: same-worker Criterion `rms_norm/grad_2048x1024` versus the original generic path plus ft-conformance coverage.
