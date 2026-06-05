# ft-api LOBPCG wrapper closeout

Bead: `frankentorch-gsmd`
Date: 2026-06-05
Agent: BlackThrush

## Profile-backed target

The kernel lever was already landed in `31f0e5d9`: `lobpcg_contiguous_f64`
computes top/bottom symmetric eigenpairs in `O(n^2 k)` per iteration instead of
forcing callers through full `eigh` `O(n^3)`.

Recorded rch Criterion evidence from that commit:

```text
eigh_f64_256x256 = 77.96 ms
lobpcg_f64_256x256_k8 = 16.82 ms
p50 speedup = 4.6x
```

## One lever

Expose the landed kernel through `ft-api` as `tensor_lobpcg` and
`functional_lobpcg`, returning detached no-grad `(eigenvalues[k],
eigenvectors[n,k])` tensors. No kernel arithmetic changed in this closeout.

## Score

```text
Impact 4.6 * Confidence 0.90 / Effort 1.0 = 4.14
```

Verdict: keep.

## Isomorphism proof

- Ordering preserved: yes. `largest=true` preserves the kernel's descending
  extreme-pair selection; `largest=false` preserves ascending bottom selection.
- Tie-breaking unchanged: N/A. The fixture has a well-separated spectrum.
- Floating-point preserved: yes. The wrapper delegates to the same
  `lobpcg_contiguous_f64` kernel without reordering arithmetic.
- RNG unchanged: yes. The kernel's deterministic SplitMix64 initialization is
  unchanged and the API wrapper does not touch RNG state.
- Shape/error behavior unchanged: yes. Kernel validation remains the source of
  square-matrix and storage-layout checks; the wrapper only maps outputs into
  tensor nodes.
- Golden output: `artifacts/optimization/golden_outputs/ft_api_lobpcg_frankentorch-gsmd.txt`
- Golden digest: `0xba286106f6b3237c`

## Validation

Passed:

```text
rch exec -- cargo test -p ft-api lobpcg_api_matches_eigh_extreme_pairs -- --nocapture
rch exec -- cargo test -p ft-kernel-cpu lobpcg_top_k_matches_eigh -- --nocapture
rch exec -- cargo check -p ft-api --all-targets
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
git diff --check -- crates/ft-api/src/lib.rs artifacts/optimization/2026-06-05_ft_api_lobpcg_frankentorch-gsmd.md artifacts/optimization/golden_outputs/ft_api_lobpcg_frankentorch-gsmd.txt artifacts/optimization/golden_checksums.txt .beads/issues.jsonl
```

Known repo-wide blockers, not introduced by this LOBPCG hunk:

```text
rch exec -- cargo clippy -p ft-api --all-targets -- -D warnings
  FAILED: broad existing ft-api clippy debt; first lib findings include
  needless_range_loop at src/lib.rs:2217, manual_is_multiple_of at src/lib.rs:5217,
  needless_range_loop at src/lib.rs:6708, and excessive_precision constants.

rch exec -- cargo fmt -p ft-api -- --check
  FAILED: broad existing formatting drift in crates/ft-api/src/lib.rs and ft-api benches.

timeout 360s ubs crates/ft-api/src/lib.rs artifacts/optimization/2026-06-05_ft_api_lobpcg_frankentorch-gsmd.md artifacts/optimization/golden_outputs/ft_api_lobpcg_frankentorch-gsmd.txt artifacts/optimization/golden_checksums.txt .beads/issues.jsonl
  FAILED after 321s with existing high-volume ft-api findings. UBS shadow cargo
  check/test/clippy/fmt sections were clean, but repository UBS policy still exits
  nonzero on long-standing unwrap/panic/indexing/finding classes in ft-api.
```
