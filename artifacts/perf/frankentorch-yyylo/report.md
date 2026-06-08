# frankentorch-yyylo SVD normal-equations trial

## Target

- Bead: `frankentorch-yyylo`
- Benchmark: `svd_f64_256x256` in `crates/ft-kernel-cpu/benches/linalg_bench.rs`
- Baseline, `vmi1293453`: `[1.6536 s 1.6664 s 1.6803 s]`
- Fresh baseline, `vmi1149989`: `[1.5532 s 1.5776 s 1.6056 s]`

## Profile Split

- `svdvals_f64_256x128`, `vmi1227854`: `[6.5906 ms 7.0747 ms 7.6529 ms]`
- `svd_f64_256x128`, `fmd`: `[32.082 ms 32.313 ms 32.603 ms]`
- `eigh_f64_256x256`, `vmi1227854`: `[10.704 ms 10.954 ms 11.252 ms]`

The values-only path and symmetric eigensolver are not the dominant cost. The
square full-SVD target is dominated by vector accumulation / rank-deficient
convergence behavior in the Golub-Reinsch bidiagonal QR path.

## Lever Tested

One source lever was tested and rejected:

- Build reduced SVD through the symmetric normal-equations problem
  `A^T A = V diag(s^2) V^T`.
- Recover `U = A V / sigma`.
- Guard with finite checks, sorted singular values, orthogonality checks, and
  reconstruction residual checks.
- Variant 1 rejected clustered / zero singular values and fell back to
  Golub-Reinsch on the target matrix.
- Variant 2 retained zero singular values and completed U with deterministic
  Gram-Schmidt; it preserved focused tests but regressed the target benchmark.

## Behavior Proof

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu svd_ -- --nocapture`
  passed during the trial:
  - `vmi1227854`: 13 passed, 0 failed
  - `vmi1156319`: 13 passed, 0 failed
  - `ovh-b`: 13 passed, 0 failed
- Ordering / tie-breaking: candidate sorted singular values descending with
  deterministic eigenvector sign canonicalization; clustered/zero spectra were
  validated through reconstruction and orthogonality before use.
- Floating point: candidate changed the algorithm and therefore did not claim
  bit-identical U/V output. It was eligible only if semantic SVD invariants and
  the perf gate both passed.
- RNG: none.
- Final source state: the candidate source hunk was removed. `git diff --
  crates/ft-kernel-cpu/src/lib.rs` is empty for this session's SVD lever.

## Rebench

- Variant 1, exact target, `vmi1227854`: `[1.5648 s 1.5822 s 1.5959 s]`
  - The guard fell back to Golub-Reinsch, so there was no target win.
- Variant 2, rank-deficient completion, `vmi1153651`: `[4.7615 s 4.8965 s 5.0429 s]`
  - Regressed the target by roughly 3x vs the baseline family.
- Variant 2 after one-pass completion, `ovh-b`: `[1.9990 s 2.0268 s 2.0546 s]`
  - Still slower than the baseline family.

## Score

- Impact: `0` on the target
- Confidence: `1.0`
- Effort: `3`
- Score: `0.0`, below the `>= 2.0` keep gate
- Verdict: `REJECTED`; no source change kept.

## Next Primitive

Do not repeat the normal-equations shortcut for this bead. The next profile-backed
SVD attack should be one of:

1. Divide-and-conquer bidiagonal SVD (`dbdsdc`-class) with exact fallback and a
   reconstruction/orthogonality proof bundle.
2. BLAS-3 back-transformation: accumulate Givens rotations into blocks and apply
   to U/V through the existing safe-Rust GEMM layer, with strict-mode fallback
   for bit-sensitive cases.
3. Deflation/rank-revealing bidiagonal QR for clustered or zero singular values,
   preserving the current reduced-SVD contract.
