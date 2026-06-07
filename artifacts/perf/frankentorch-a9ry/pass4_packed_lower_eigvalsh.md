# frankentorch-a9ry pass 4: packed lower values-only tridiagonalization

Target: `ft-kernel-cpu` values-only `eigvalsh_f64_256x256`, where the shared Householder tridiagonalization remains the dominant residual.

Fresh same-worker baseline:

- Worker: `ts2`
- `eigvalsh_f64_256x256`: `[10.543 ms 10.576 ms 10.610 ms]`

Alien primitive:

- Different memory layout: store the values-only symmetric matrix as a packed lower triangle.
- This is the safe-Rust LAPACK-family direction for symmetric eigensolvers: separate the values-only path from the full eigenvector path and reduce memory traffic before the larger blocked/D&C primitive.

Lever:

- Add `lower_packed_index(row, col)`.
- Change only `eigh_tred2_values_only` and `eigvalsh_contiguous_f64` to use `n * (n + 1) / 2` packed lower storage.
- Preserve the Householder scale loop, normalization loop, dot-product order, `f` accumulation, rank-2 update expressions, QL iteration, and final `total_cmp` sort.
- Full `eigh` keeps the existing square workspace and back-transform path.

Behavior proof:

- Ordering and tie-breaking: unchanged. `eigvalsh_contiguous_f64` still sorts `d` by `f64::total_cmp`.
- Floating point: unchanged for the values path. The packed storage addresses the same lower-triangle entries in the same loop order; only unused upper-triangle/reflector-column storage is removed.
- RNG: not used.
- Golden before SHA-256: `1870e56ea935f9cc895b24d878db52fe341dc2b195c00656faa38b2db97ac458`
- Golden after SHA-256: `1870e56ea935f9cc895b24d878db52fe341dc2b195c00656faa38b2db97ac458`
- `cmp artifacts/perf/frankentorch-a9ry/eigvalsh_packed_lower_before.txt artifacts/perf/frankentorch-a9ry/eigvalsh_packed_lower_after.txt` passed.

After benchmark:

- Worker: `ts2`
- `eigvalsh_f64_256x256`: `[9.5529 ms 9.5769 ms 9.6087 ms]`
- Values-only median speedup: `10.576 / 9.5769 = 1.104x`.
- Score: `Impact 3.0 x Confidence 4.0 / Effort 2.0 = 6.0`, keep.

Verification:

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigvalsh_f64_256x256 --warm-up-time 1 --measurement-time 5 --sample-size 20`: after row above.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- env FT_EIGVALSH_GOLDEN=1 cargo run -p ft-kernel-cpu --example eigh_golden`: generated the after golden.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo check -p ft-kernel-cpu --all-targets`: passed.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo test -p ft-kernel-cpu eigvalsh -- --nocapture`: passed `eigvalsh_matches_eigh`.

Residual and next primitive:

- The values-only row is now `9.5769 ms` median on `ts2`.
- The next primitive remains the larger safe-Rust LAPACK-class blocked tridiagonalization plus tridiagonal D&C/secular merge, with exact-order fallback where the floating-point proof gate cannot be satisfied.
