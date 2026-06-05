# frankentorch-ncwz pass 3: row-contiguous Householder back-transform

Target: `ft-kernel-cpu` full-vector `eigh_f64_256x256`, Householder eigenvector back-transform in `eigh_tred2`.

Baseline:

- Current pass-1 state on `ts1`: `49.338 ms` median for `eigh_f64_256x256`.
- Command: `rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigh_f64_256x256`

Lever:

- Compute the same projection scalars `g_j = sum_k row_i[k] * z[k, j]` in the same `j` then `k` order.
- Store those projection scalars, then apply the independent cell updates row-contiguously:
  - before: for each `j`, update `z[k, j]` down the column.
  - after: for each `k`, update row slice `z[k, 0..i]`.
- Each cell still receives exactly `old - g_j * reflector_col_k`; only the order of independent writes changes.

Behavior proof:

- Golden SHA-256 before: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- Golden SHA-256 after: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- `cmp artifacts/perf/frankentorch-ncwz/eigh_golden_before.txt artifacts/perf/frankentorch-ncwz/eigh_golden_pass3_after.txt` passed.

After benchmark:

- Worker: `ts1`
- After median: `35.423 ms`
- Delta: `28.9%` faster, `1.393x`.
- Score: `Impact 3.0 x Confidence 5 / Effort 2 = 7.5`, keep.

Verification:

- `rch exec -- cargo check -p ft-kernel-cpu --all-targets`
- `rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`
- `rch exec -- cargo test -p ft-kernel-cpu`

Residual:

- Full-vector `eigh` still carries `tql2` O(n^3) rotation accumulation. The next algorithmic target remains tridiagonal divide-and-conquer / secular-equation merge, but this pass removes a real cache-local back-transform bottleneck without changing bits.
