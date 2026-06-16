## frankentorch-kgs4.103 proof

Target: `batch_norm/grad_1d_8192x1024` from the focused ft-api training reprofile on RCH worker `vmi1149989`.

### Baseline

Artifact: `artifacts/perf/frankentorch-ftapi-train-reprofile-20260616/baseline_train_hotspots.log`

- `batch_norm/grad_1d_8192x1024`: `[950.83 ms 1.0076 s 1.0625 s]`
- `batch_norm/grad_train_32x256x28x28`: `[615.06 ms 659.70 ms 715.49 ms]`

### Lever

One lever: specialize `ft_kernel_cpu::batch_norm_backward_f64` when `spatial == 1`.

The generic path reduced by channel but wrote `dx` with `par_chunks_mut(1)`, creating one parallel work item per scalar for BatchNorm1d `[N, C]`. The new path keeps the exact per-channel reduction order (`n` increasing for each channel), precomputes the same `rstd[c]`, and writes `dx` by `[N, C]` row chunks. Each output element remains independent.

### Isomorphism

- Ordering: per-channel `dweight`/`dbias` accumulation order is unchanged (`n = 0..batch`).
- Tie-breaking: none.
- Floating point: each reduction uses the same expression order. `dx` uses the same formula; `rstd[c]` is computed from the same expression and reused instead of recomputed per element.
- RNG: none in the kernel. The benchmark setup RNG is outside the optimized kernel.

Golden output:

- Baseline SHA: `3985a428649677e0e6150555aef5038e1a449a7b35413a5d5cf0ebc777ac9843`
- Candidate SHA: `3985a428649677e0e6150555aef5038e1a449a7b35413a5d5cf0ebc777ac9843`
- `diff -u pass1_baseline_batch_norm1d_golden_lines.txt pass2_candidate_batch_norm1d_golden_lines.txt`: no differences.

### Candidate

Artifact: `artifacts/perf/frankentorch-kgs4.103/pass3_candidate_batch_norm_rebench_vmi1149989.log`

- `batch_norm/grad_1d_8192x1024`: `[445.94 ms 472.43 ms 501.42 ms]`
- `batch_norm/grad_train_32x256x28x28`: `[435.48 ms 477.00 ms 518.52 ms]`

Speedup:

- 1D target mean: `1.0076 s / 472.43 ms = 2.13x`
- Adjacent 2D guard mean: `659.70 ms / 477.00 ms = 1.38x`

Score: `Impact 4.0 x Confidence 0.95 / Effort 1.5 = 2.53`, keep.

### Validation

- `cargo test -j 1 -p ft-api functional_batch_norm1d_grad -- --nocapture` on `vmi1149989`: passed 2/2 focused tests.
- `cargo check -j 1 -p ft-api --lib` on `vmi1149989`: passed.
- `git diff --check`: passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs` with a 180s cap: completed, 0 critical issues, existing broad warnings.
- `ubs crates/ft-api/src/lib.rs` with a 180s cap: timed out while scanning the very large API file; no findings emitted before timeout. The API change is test-only and is covered by the focused RCH test above.
