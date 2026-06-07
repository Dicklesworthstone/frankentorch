# frankentorch-c6uw pass 8: direct reflector-column read in `eigh` back-transform

## Target

Current full-vector `eigh_f64_256x256` residual after the packed full-vector
tridiagonal reduction keep. The larger no-gaps target remains blocked
symmetric tridiagonalization plus tridiagonal divide-and-conquer, but this pass
tested one exact EISPACK-stream sublever before starting the FP-reassociating
rewrite.

Same-worker baseline on `ts2`:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- \
  cargo bench -p ft-kernel-cpu --bench linalg_bench -- \
  'eigh_f64_256x256|eigvalsh_f64_256x256' \
  --warm-up-time 1 --measurement-time 5 --sample-size 20
```

- `eigh_f64_256x256`: `[15.920 ms 16.299 ms 16.661 ms]`
- `eigvalsh_f64_256x256`: `[9.8632 ms 9.9481 ms 10.017 ms]`

## Change

One lever: remove the temporary `reflector_col` vector copy in
`eigh_tred2_backtransform`.

Before the update loop, the code copied `z[k, i]` for `k < i` into a temporary
vector. The row update mutates only each row prefix `0..i`, not column `i`, so
the same reflector value can be read directly from `previous_rows[k * n + i]`
immediately before the prefix update.

## Isomorphism Proof

- Ordering preserved: yes. Final eigenpair sorting still uses `f64::total_cmp`
  on `(eigenvalue, old_col)` pairs and writes columns through the same
  permutation.
- Tie-breaking unchanged: yes. Pair construction order remains `0..n`; stable
  sort behavior is untouched.
- Floating-point: bit-identical. The projection accumulation and row update
  expressions execute in the same `i`, `k`, `j` order. Only the storage source
  for the reflector scalar changed from a copied vector to the unchanged column
  slot.
- RNG seeds: N/A.
- Golden outputs: clean full-eigh fixture before and after both produced
  SHA-256 `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`;
  `cmp` passed.

## Benchmark

After candidate on `ts2`:

- `eigh_f64_256x256`: `[15.397 ms 15.511 ms 15.629 ms]`
- `eigvalsh_f64_256x256`: `[9.6501 ms 10.299 ms 11.353 ms]` with one high
  severe outlier

Full-vector median speedup: `16.299 / 15.511 = 1.051x`.

`eigvalsh` is not routed through the changed back-transform; its row is retained
as a same-binary control and was noisy on the after run.

## Gate

Score: `Impact 2.0 x Confidence 3.5 / Effort 1.0 = 7.0`; keep.

Verification:

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo check -p ft-kernel-cpu --all-targets`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo test -p ft-kernel-cpu eigvalsh_matches_eigh -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo test -p ft-kernel-cpu eigh_tred2_tql2_orthonormal_and_reconstructs_24x24 -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`

Format note: `cargo fmt -p ft-kernel-cpu --check` currently fails on existing
format drift outside this lever (`conv2d_forward_f32` call wrapping and
previous packed-tridiag helper call wrapping). This pass did not bulk-format
those unrelated lines.

Next primitive: follow-up bead `frankentorch-rd1s` tracks the deeper
safe-Rust LAPACK-class blocked `dsytrd` panel / compact-WY / BLAS-3 trailing
rank-2k update plus tridiagonal D&C/secular merge.
