# frankentorch-ncwz pass 2: tql2 row-chunk rotation rejected

Target: `ft-kernel-cpu` full-vector `eigh_f64_256x256`, residual `tql2` eigenvector rotation accumulation.

Candidate lever:

- Replace indexed row traversal in `eigh_tql2` rotation accumulation with `chunks_exact_mut(n)`.
- Cache the left/right row values before writing the rotated columns.
- Preserve the formulas and operation order:
  - `z[k, i + 1] = s * z[k, i] + c * old_z[k, i + 1]`
  - `z[k, i] = c * z[k, i] - s * old_z[k, i + 1]`

Behavior proof:

- Golden SHA-256: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- `cmp artifacts/perf/frankentorch-ncwz/eigh_golden_before.txt artifacts/perf/frankentorch-ncwz/eigh_golden_pass2_after.txt` passed.

Benchmark evidence:

- Pass-1 same-worker `ts2` baseline: `74.926 ms` median.
- Candidate `ts2` result: `79.227 ms` median.
- Delta: `5.65%` slower.
- Score: rejected, below keep gate.

Disposition:

- Reverted the code change manually; the working tree has no `ft-kernel-cpu/src/lib.rs` diff from pass 1.
- This is a signal to leave row-loop micro-tuning and attack the deeper primitive.

Next primitive:

- Tridiagonal divide-and-conquer eigensolver for full-vector `eigh`: replace `tql2`'s O(n^3) rotation accumulation with secular-equation rank-1 merge plus deflation.
- Target ratio: at least `2x` on full-vector `eigh_f64_256x256`, with exact ordering/tie policy and signed-zero behavior explicitly pinned by golden fixtures before any performance claim.
