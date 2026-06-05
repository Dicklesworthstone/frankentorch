# frankentorch-ncwz pass 4: row-contiguous Householder projections

Target: `ft-kernel-cpu` full-vector `eigh_f64_256x256`, Householder eigenvector back-transform in `eigh_tred2`.

Baseline:

- Pass-3 code restored locally for a same-worker baseline on `ts1`: `34.758 ms` median for `eigh_f64_256x256`.
- Command: `rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigh_f64_256x256`

Lever:

- Keep the same projection scalar definition `g_j = sum_k row_i[k] * z[k, j]`.
- Before: compute one projection at a time, reading `z[k, j]` down columns.
- After: visit each previous row once and accumulate every independent `g_j` from that row slice.
- For each fixed `j`, floating-point additions still occur in ascending `k` order, so each projection scalar has the same arithmetic order as before.

Behavior proof:

- Ordering and tie-breaking: unchanged. This path has no data-dependent sort or tie-breaking change.
- Floating-point: for each projection `j`, products and additions are evaluated in the same `k = 0..i` order; only independent projection streams are interleaved.
- RNG: not used.
- Golden SHA-256 before: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- Golden SHA-256 after: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- `cmp artifacts/perf/frankentorch-ncwz/eigh_golden_before.txt artifacts/perf/frankentorch-ncwz/eigh_golden_pass4_after.txt` passed.

After benchmark:

- Worker: `ts1`
- After median: `20.596 ms`
- Delta: `40.7%` faster, `1.688x`.
- Score: `Impact 4.0 x Confidence 5 / Effort 2 = 10.0`, keep.

Verification:

- `rch exec -- cargo check -p ft-kernel-cpu --all-targets`: passed.
- `rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`: passed.
- `rch exec -- cargo test -p ft-kernel-cpu`: passed, 397 tests.
- `ubs crates/ft-kernel-cpu/src/lib.rs artifacts/perf/frankentorch-ncwz/pass4_eigh_tred2_row_projections.md artifacts/perf/frankentorch-ncwz/eigh_golden_pass4_after.txt .skill-loop-progress.md`: exit 0, 0 critical findings. Broad warning inventory in `ft-kernel-cpu/src/lib.rs` remains outside this lever.
- `rch exec -- cargo fmt -p ft-kernel-cpu --check`: failed on pre-existing formatting drift in GEMM/conv/eig/SVD sections outside this pass-4 loop; not bulk-formatted in this one-lever commit.

Residual:

- The remaining full-vector `eigh` cost is now concentrated in the tridiagonal eigensolver and eigenvector merge/rotation path. The next deeper primitive is tridiagonal divide-and-conquer with a safe-Rust secular-equation merge, not further tuning of the rejected `tql2` rotation loop.
