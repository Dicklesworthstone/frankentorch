# frankentorch-rg1n pass 1: row-sliced Householder tridiagonal reduction

Target: `ft-kernel-cpu` full-vector `eigh_f64_256x256`, shared Householder tridiagonalization in `eigh_tred2_reduce`.

Profile and baseline:

- Prior profile in `frankentorch-rg1n`: after vector-layout passes, `eigh_f64_256x256` median `11.925 ms`, `eigvalsh_f64_256x256` median `7.964 ms`.
- Fresh same-worker baseline on `ts2`:
  - `eigh_f64_256x256`: `[16.717 ms 16.761 ms 16.808 ms]`
  - `eigvalsh_f64_256x256`: `[11.866 ms 11.911 ms 11.958 ms]`
- The split leaves the shared reduction/QL path as the larger residual, with full-vector-only work around `4.85 ms` on this worker.

Alien primitive selection:

- Canonical graveyard entry 9.6 points to communication-avoiding dense linear algebra and BLAS-3 inner kernels as the deeper direction.
- Family 34 numerical linear algebra proof requirements demand exact decomposition behavior, orthogonality/reconstruction checks, and conditioning/error ledgers.
- A full blocked `dsytrd` or tridiagonal D&C/secular merge would reassociate floating-point work, so this pass took the exact-arithmetic sublever first: keep the EISPACK operation stream, but make the active Householder row explicit as a slice.

Lever:

- In `eigh_tred2_reduce`, split `z` at the active row once per Householder step.
- Read/write the active reflector row through `row_i`, and previous rows through `previous_rows`.
- Preserve every `j` loop, `k` loop, dot-product order, rank-2 update expression, and `d`/`e` update.

Behavior proof:

- Ordering and tie-breaking: unchanged. `eigh_contiguous_f64` still sorts `(value, old_col)` pairs with `f64::total_cmp`, and `eigvalsh_contiguous_f64` still sorts `d` with `total_cmp`.
- Floating point: bit-identical. The Householder scale loop, normalization loop, `gg` dot-product loops, `f` accumulation, `hh`, and trailing update use the same expressions in the same order. Only address calculation moved from repeated `z[i * n + k]` indexing to slices over the same storage.
- RNG: not used.
- Golden before SHA-256: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- Golden after SHA-256: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- `cmp artifacts/perf/frankentorch-rg1n/eigh_golden_before.txt artifacts/perf/frankentorch-rg1n/eigh_golden_after.txt` passed.

After benchmark:

- Worker: `ts2`
- `eigh_f64_256x256`: `[15.610 ms 15.657 ms 15.706 ms]`
- `eigvalsh_f64_256x256`: `[10.536 ms 10.563 ms 10.591 ms]`
- Full `eigh` median speedup: `16.761 / 15.657 = 1.071x`.
- Values-only median speedup: `11.911 / 10.563 = 1.128x`.
- Score: `Impact 2.5 x Confidence 4.0 / Effort 1.0 = 10.0`, keep.

Verification:

- `rch exec -- cargo check -p ft-kernel-cpu --all-targets`: passed.
- `rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`: passed.
- `rch exec -- cargo test -p ft-kernel-cpu eigh -- --nocapture`: passed, 6 tests.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`: passed.
- `git diff --check -- crates/ft-kernel-cpu/src/lib.rs artifacts/perf/frankentorch-rg1n/pass1_eigh_tred2_row_slice.md artifacts/perf/frankentorch-rg1n/eigh_golden_before.txt artifacts/perf/frankentorch-rg1n/eigh_golden_after.txt artifacts/optimization/golden_checksums.txt .skill-loop-progress.md .beads/issues.jsonl`: passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs artifacts/perf/frankentorch-rg1n/pass1_eigh_tred2_row_slice.md artifacts/perf/frankentorch-rg1n/eigh_golden_before.txt artifacts/perf/frankentorch-rg1n/eigh_golden_after.txt artifacts/optimization/golden_checksums.txt .skill-loop-progress.md .beads/issues.jsonl`: exit 0, 0 critical findings. Broad warning inventory in `ft-kernel-cpu/src/lib.rs` remains outside this lever.
- `rch exec -- cargo fmt -p ft-kernel-cpu --check`: failed on pre-existing package-wide formatting drift outside the staged `eigh_tred2_reduce` hunk; this commit does not bulk-format unrelated code.

Residual and next primitive:

- The shared values path is still `10.563 ms` median on `ts2`. The next deeper primitive is a real safe-Rust LAPACK-class eigensolver step: a blocked tridiagonalization/D&C design with an explicit floating-point parity contract, eigenvalue ordering/tie ledger, eigenvector sign/orientation proof, and fallback to the current exact EISPACK path when the proof gate cannot be satisfied.
