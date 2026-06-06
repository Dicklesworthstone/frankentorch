# frankentorch-0qwa: no-grad pinv QR fast path

## Target

- Bead: `frankentorch-0qwa` (`[perf][no-gaps] ft-api pinv: QR fast path for full-rank tall/wide matrices`)
- Profile-backed hotspot: `ft-api` Criterion `lstsq/pinv_svd`, which calls `tensor_linalg_pinv` and was dominated by the no-grad SVD path for full-rank rectangular matrices.
- Primitive harvested from alien-graveyard/no-gaps routing: communication-avoiding QR family, narrowed for this pass to a safe-Rust Q-free Householder QR pseudo-inverse primitive. The lever avoids materializing full `m x m` Q and back-solves only the first `n` rows of `Q^T`.

## One Lever

Added a strictly gated no-grad fast path for contiguous f64 2-D full-rank inputs:

- Tall/full-column-rank `m >= n`: compute `A+ = R^-1 Q^T` from Householder QR.
- Wide/full-row-rank `m < n`: factor `A^T`, then transpose `(A^T)+` to `A+`.
- Empty, non-2-D, unsupported layout, rank-deficient, and near-rank-deficient inputs defer to the existing SVD path.
- Requires-grad `tensor_linalg_pinv` branch is unchanged.

## Same-Worker Benchmark

Command, baseline:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- lstsq/pinv_svd --warm-up-time 1 --measurement-time 5 --sample-size 10
```

Baseline on `ts1`:

- `lstsq/pinv_svd_256x128`: `[24.473 ms 25.694 ms 26.881 ms]`
- `lstsq/pinv_svd_512x256`: `[443.94 ms 461.21 ms 476.15 ms]`

After on `ts1`:

- `lstsq/pinv_svd_256x128`: `[12.791 ms 12.903 ms 13.021 ms]`, median speedup `1.99x`
- `lstsq/pinv_svd_512x256`: `[164.40 ms 169.07 ms 172.64 ms]`, median speedup `2.73x`

Post-rebase over `63ca3fa1` (`multi_dot` DP), final after on `ts1`:

- `lstsq/pinv_svd_256x128`: `[12.873 ms 13.002 ms 13.122 ms]`, median speedup `1.98x`
- `lstsq/pinv_svd_512x256`: `[162.82 ms 164.75 ms 166.34 ms]`, median speedup `2.80x`

Score: `Impact 4.0 x Confidence 0.95 / Effort 1.4 = 2.71`, keep.

## Isomorphism Proof

- Shape/order: output remains row-major shape `[n, m]`. Tall path writes `[n, m]` directly; wide path transposes the pseudo-inverse of `A^T` into the same `[n, m]` shape.
- Branch contract: only no-grad contiguous f64 2-D inputs enter the QR fast path. Requires-grad stays on the existing autograd/SVD route, preserving gradient graph and DAC behavior.
- Rank/tie contract: QR fast path accepts only conservative full-rank cases. Duplicate columns/rows and near-zero pivots return `None` and keep the existing SVD fallback. False fallback is safe; false fast path is guarded against by the relative diagonal tolerance.
- Floating point: the fast path is not bit-identical to SVD, so proof is tolerance based. Known diagonal embedding tests assert exact expected values within `1e-12`; fallback tests preserve SVD behavior for rank-deficient cases; full pinv API tests preserve prior roundtrip/gradient coverage.
- RNG: no RNG call or seed/order is changed. The benchmark input generation remains the existing fixed Criterion path.
- Ordering/tie-breaking: no external sorting or tie logic is introduced. Internal Householder sign choice is deterministic from the current column norm and first element.

Subagent proof review (`Hooke`) agreed the safe lever is a pinv-specific QR helper called only before SVD in the no-grad branch, with conservative fallback for rank-deficient and unsupported cases.

## Validation

- `cargo check -p ft-kernel-cpu -p ft-api --all-targets` via `rch` on `ts1`: passed (`check_pinv_qr_after_clippy_fix.txt`).
- `cargo test -p ft-kernel-cpu pinv_qr -- --nocapture` via `rch` on `ts1`: passed, `2 passed` (`test_kernel_pinv_qr_after_clippy_fix.txt`).
- `cargo test -p ft-api pinv -- --nocapture` via `rch` on `ts1`: passed, `9 passed` (`test_api_pinv_all_after_clippy_fix.txt`).
- Post-rebase `cargo check -p ft-kernel-cpu -p ft-api --all-targets` via `rch` on `ts1`: passed (`post_rebase_check_pinv_qr.txt`).
- Post-rebase `cargo test -p ft-kernel-cpu pinv_qr -- --nocapture` via `rch` on `ts1`: passed, `2 passed` (`post_rebase_test_kernel_pinv_qr.txt`).
- Post-rebase `cargo test -p ft-api pinv -- --nocapture` via `rch` on `ts1`: passed, `9 passed` (`post_rebase_test_api_pinv_all.txt`).
- Post-rebase `cargo bench -p ft-api --bench ops_bench -- lstsq/pinv_svd --warm-up-time 1 --measurement-time 5 --sample-size 10` via `rch` on `ts1`: passed (`post_rebase_after_pinv_qr.txt`).
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`: passed (`golden_sha256_check.txt`).
- `git diff --check`: passed (`git_diff_check.txt`).
- `cargo clippy -p ft-kernel-cpu --all-targets --no-deps -- -D warnings` via `rch` on `ts1`: passed (`clippy_ft_kernel_cpu_all_targets.txt`).
- `cargo clippy -p ft-api --lib --no-deps -- -D warnings` and the broader two-crate `--all-targets` clippy gate were attempted and recorded. They fail on the existing broad `ft-api` lint backlog (`183` library errors, `206` test-target errors), with no pinv-specific diagnostics found. Those are outside this one-lever bead.
- `ubs crates/ft-kernel-cpu/src/lib.rs crates/ft-api/src/lib.rs .skill-loop-progress.md artifacts/perf/frankentorch-0qwa/report.md` was attempted and terminated after roughly three minutes without actionable findings; partial log and timeout note are retained in `ubs_changed_surface.txt` and `ubs_timeout_note.txt`.

## Next Profile Route

After this closeout, re-run `br ready --json`/profiling and route to the next top `[perf]` bead. For linalg residuals, the next structural primitive is a deeper blocked/TSQR QR-family implementation or batched panel solve only after a fresh profile-backed target identifies it.
