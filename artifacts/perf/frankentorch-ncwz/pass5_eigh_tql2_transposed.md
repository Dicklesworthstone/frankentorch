# frankentorch-ncwz pass 5: transposed tql2 eigenvector rotations

Target: `ft-kernel-cpu` full-vector `eigh_f64_256x256`, implicit-shift QL eigenvector rotation accumulation in `eigh_tql2`.

Profile:

- After pass 4, same-run split on `ts2`: full `eigh_f64_256x256` median `31.883 ms`; values-only `eigvalsh_f64_256x256` median `11.380 ms`.
- That left `20.5 ms` in vector work skipped by the values-only path, so pass 5 targeted the full-vector rotation layout.

Baseline:

- Current pass-4 state on `ts1`: `20.596 ms` median for `eigh_f64_256x256`.
- Command: `rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigh_f64_256x256`

Lever:

- Transpose the accumulated Householder eigenvector matrix after `eigh_tred2`.
- Run the same QL rotation stream against `zt[col * n + row]`, so each two-column eigenvector rotation updates contiguous slices instead of striding through row-major columns.
- Copy sorted eigenvectors back to the public row-major, columns-as-eigenvectors output layout from the transposed storage.

Behavior proof:

- Ordering and tie-breaking: the eigenvalue sort and stable `total_cmp` ordering are unchanged.
- Floating-point: the QL `d`/`e` update stream is unchanged. For every rotation and row `k`, each eigenvector component evaluates the same `s * left + c * right` and `c * left - s * right` expressions in the same loop order; only the storage address mapping changes.
- RNG: not used.
- Golden SHA-256 before: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- Golden SHA-256 after: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- `cmp artifacts/perf/frankentorch-ncwz/eigh_golden_before.txt artifacts/perf/frankentorch-ncwz/eigh_golden_pass5_after.txt` passed.

After benchmark:

- Worker: `ts1`
- After median: `11.854 ms`
- Delta: `42.4%` faster, `1.737x`.
- Score: `Impact 4.0 x Confidence 5 / Effort 3 = 6.7`, keep.

Verification:

- `rch exec -- cargo check -p ft-kernel-cpu --all-targets`: passed.
- `rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`: passed.
- `rch exec -- cargo test -p ft-kernel-cpu`: passed, 397 tests.
- `ubs crates/ft-kernel-cpu/src/lib.rs artifacts/perf/frankentorch-ncwz/pass5_eigh_tql2_transposed.md artifacts/perf/frankentorch-ncwz/eigh_golden_pass5_after.txt .skill-loop-progress.md`: exit 0, 0 critical findings. Broad warning inventory in `ft-kernel-cpu/src/lib.rs` remains outside this lever.
- `rch exec -- cargo fmt -p ft-kernel-cpu --check`: failed on pre-existing formatting drift in GEMM/conv/eig/SVD sections outside this pass-5 layout lever; not bulk-formatted in this one-lever commit.

Residual:

- Full-vector `eigh` is now close to the current values-only split for this fixture. The next deeper primitive is not more strided-loop tuning: it is blocked tridiagonalization plus tridiagonal divide-and-conquer/secular merge so the solver shifts to a BLAS-3 and communication-avoiding structure.
