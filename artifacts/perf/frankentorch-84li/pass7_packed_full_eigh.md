# frankentorch-84li pass 7: packed full-vector `eigh` tridiagonal reduction

## Target

Profile-backed residual: full-vector `eigh_f64_256x256` stayed around `16.746 ms` on `ts2` after the packed values-only `eigvalsh` keep. The remaining full path still reduced through a full `n x n` work matrix before the eigenvector back-transform and transposed QL rotation stream.

Baseline same-worker Criterion:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- \
  cargo bench -p ft-kernel-cpu --bench linalg_bench -- \
  'eigh_f64_256x256|eigvalsh_f64_256x256' \
  --warm-up-time 1 --measurement-time 5 --sample-size 20
```

- `eigh_f64_256x256`: `[16.699 ms 16.746 ms 16.808 ms]`
- `eigvalsh_f64_256x256`: `[9.4280 ms 9.4513 ms 9.4802 ms]`

## Change

One lever: route full-vector `eigh_contiguous_f64` through a packed-lower Householder tridiagonal reduction, while storing the scaled upper reflector columns separately. After reduction, reconstruct the exact full work matrix and run the existing eigenvector back-transform, transposed QL rotation stream, and final `total_cmp` eigenpair ordering.

This is the same memory-layout primitive that won for values-only `eigvalsh`, extended to full `eigh` without changing the rotation stream or final orientation logic.

## Isomorphism Proof

- Ordering preserved: yes. The final eigenpair vector is still sorted by `f64::total_cmp`, and the same `old_col` permutation drives eigenvector columns.
- Tie-breaking unchanged: yes. Pair construction order remains `0..n`, and stable `sort_by` semantics are untouched.
- Floating-point: the Householder reduction performs the same scale, normalization, dot, `e[j]`, `f`, `hh`, and trailing-update operations in the same loop order. The layout changes only where values are stored: lower-triangle entries live in packed storage and `z[j,i] = z[i,j] / h` lives in `scaled_reflectors`; reconstruction restores the full matrix before the existing back-transform.
- RNG seeds: N/A.
- Golden outputs: clean full-eigh fixture before and after both produced SHA-256 `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`.

## Benchmark

After candidate, before dead-code cleanup:

- `eigh_f64_256x256`: `[15.227 ms 15.282 ms 15.331 ms]`
- `eigvalsh_f64_256x256`: `[9.4869 ms 9.5168 ms 9.5491 ms]`

Final kept tree after formatting and removing unused old reducer code:

- `eigh_f64_256x256`: `[15.285 ms 15.309 ms 15.342 ms]`
- `eigvalsh_f64_256x256`: `[9.4969 ms 9.5300 ms 9.5609 ms]`

Full-vector median speedup: `16.746 / 15.309 = 1.094x`.

`eigvalsh` is not routed through the changed full-vector function; its row is retained as a control and shows ordinary same-run noise/code-layout drift.

## Gate

Score: `Impact 4.0 x Confidence 4.0 / Effort 2.5 = 6.4`; keep.

Verification:

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo check -p ft-kernel-cpu --all-targets`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo test -p ft-kernel-cpu eigvalsh_matches_eigh -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo test -p ft-kernel-cpu eigh_tred2_tql2_orthonormal_and_reconstructs_24x24 -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`

Next primitive after reprofile: blocked symmetric tridiagonalization (`dsytrd`-style panel plus compact-WY / BLAS-3 trailing update) and tridiagonal divide-and-conquer / secular merge with deterministic deflation and sign-orientation proof.
