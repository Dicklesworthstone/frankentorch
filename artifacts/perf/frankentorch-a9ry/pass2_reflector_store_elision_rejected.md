# frankentorch-a9ry pass 2: reject values-only reflector-column elision

Target: `ft-kernel-cpu` symmetric eigensolver residual after the rg1n row-slice keep.

Fresh same-worker baseline:

- Worker: `ts2`
- `eigh_f64_256x256`: `[15.510 ms 15.579 ms 15.645 ms]`
- `eigvalsh_f64_256x256`: `[10.543 ms 10.576 ms 10.610 ms]`

Candidate lever:

- Specialize `eigh_tred2_reduce` for the full-vector and values-only callers.
- In the values-only caller, skip storing `previous_rows[j * n + i] = row_i[j] / h` because the later eigenvector back-transform is not run.
- Keep the Householder scale loop, normalization loop, dot-product order, `f` accumulation, rank-2 update expressions, QL iteration, sorting, and all returned values unchanged.

Behavior proof:

- Ordering and tie-breaking: unchanged. Full `eigh` still sorts `(value, old_col)` pairs by `f64::total_cmp`; values-only `eigvalsh` still sorts `d` by `total_cmp`.
- Floating point: golden output is bit-identical. The skipped stores are not read by the values-only path after tridiagonal reduction; the full path retains them.
- RNG: not used.
- Golden before SHA-256: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- Golden after SHA-256: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- `cmp artifacts/perf/frankentorch-a9ry/eigh_golden_before.txt artifacts/perf/frankentorch-a9ry/eigh_golden_after.txt` passed.

After benchmark:

- Worker: `ts2`
- `eigh_f64_256x256`: `[15.694 ms 15.715 ms 15.755 ms]`
- `eigvalsh_f64_256x256`: `[10.537 ms 10.563 ms 10.583 ms]`
- Full `eigh` median speedup: `15.579 / 15.715 = 0.991x` (regression).
- Values-only median speedup: `10.576 / 10.563 = 1.001x` (noise-scale).
- Score: below `2.0`; reject.

Verdict:

- Source hunk removed.
- This is the micro-tuning trigger from the no-ceiling directive. The next pass must attack the larger safe-Rust LAPACK-class primitive: blocked symmetric tridiagonalization (`dsytrd`-style panel plus compact WY/BLAS-3 trailing update) and/or tridiagonal divide-and-conquer/secular merge with explicit floating-point parity ledger and fallback to the current exact EISPACK path where required.
