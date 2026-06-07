# frankentorch-a9ry pass 3: reject transposed tql2 row-slice rotations

Target: `ft-kernel-cpu` full-vector `eigh_f64_256x256`, specifically the transposed eigenvector rotation stream in `eigh_tql2_transposed`.

Fresh same-worker baseline:

- Worker: `ts2`
- `eigh_f64_256x256`: `[15.510 ms 15.579 ms 15.645 ms]`

Candidate lever:

- Split `zt` into two mutable transposed eigenvector rows for each QL rotation.
- Replace repeated `zt[col_next + k]` / `zt[col_i + k]` indexing with row slices.
- Preserve the `k` loop order and the exact rotation expressions:
  - `row_next[k] = s * left + c * f`
  - `row_i[k] = c * left - s * f`

Behavior proof:

- Ordering and tie-breaking: unchanged. `eigh_contiguous_f64` still sorts `(value, old_col)` pairs by `f64::total_cmp`.
- Floating point: bit-identical golden. The rotation stream, per-`k` order, and arithmetic expressions are unchanged; only address formation changed.
- RNG: not used.
- Golden before SHA-256: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- Golden after SHA-256: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- `cmp artifacts/perf/frankentorch-a9ry/eigh_tql2_row_slices_before.txt artifacts/perf/frankentorch-a9ry/eigh_tql2_row_slices_after.txt` passed.

After benchmark:

- Worker: `ts2`
- `eigh_f64_256x256`: `[16.166 ms 16.222 ms 16.272 ms]`
- Full `eigh` median speedup: `15.579 / 16.222 = 0.960x` (regression).
- Score: below `2.0`; reject.

Verdict:

- Source hunk removed.
- This confirms the current full-vector residual should not be attacked with more row/index micro-tuning. The next pass is the larger alien primitive: safe-Rust LAPACK-class blocked symmetric tridiagonalization plus tridiagonal divide-and-conquer/secular merge, with an explicit floating-point parity ledger and exact EISPACK fallback when the proof gate cannot be satisfied.
