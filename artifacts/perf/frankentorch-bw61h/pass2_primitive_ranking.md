# frankentorch-bw61h pass 2 primitive ranking

## Target

Measured floor: non-symmetric `eig` / `eigvals` still spends the hot path in the
scalar Francis QR sweep stream inside `eig_francis_schur_traced`.

Pass-1 baseline/proof:

- `hz1` pinned Criterion:
  - `eigvals_f64_256x256`: `[45.990 ms 47.286 ms 48.538 ms]`
  - `eig_f64_256x256`: `[77.037 ms 78.321 ms 79.717 ms]`
- `eig_timing_probe` n=256: `sweeps=319`, `defl1=14`, `defl2=121`,
  `fallback=0`, `exceptional=0`
- `eig_timing_probe` n=1024: `sweeps=1132`, `defl1=18`, `defl2=503`,
  `fallback=0`, `exceptional=0`
- Strict `eigvals_golden` stdout SHA:
  `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`

## Source lineage

- No-gaps directive: close safe-Rust BLAS/LAPACK-class gaps with one profiled
  lever, proof, and same-worker rebench.
- Graveyard section 9.6 communication-avoiding algorithms: real target is fewer
  memory movements by batching dense submatrix work into BLAS-3-like kernels.
- Current code: production still uses scalar row/column loops in the Francis
  bulge chase. The shadow proof has a same-order tiled replay, but does not prove
  reordered grouped/far-update production dispatch.

## Ranking

| Rank | Candidate | Impact | Confidence | Effort | Score | Decision |
|---|---|---:|---:|---:|---:|---|
| 1 | Same-order production tiled sweep helper: factor the existing row and column modifications into a fixed-size tiled helper, preserving exact `j` and `i` order inside each tile. | 1 | 4 | 2 | 2.0 | Safe but borderline; attempt only as fast falsification. |
| 2 | Private grouped/blocked shadow ledger: batch far-field updates in a proof-only lane before public dispatch. | 4 | 3 | 3 | 4.0 | Better EV, but needs a proof-harness bead/pass first; not a direct production source hunk now. |
| 3 | Direct grouped/far-update public Francis dispatch. | 4 | 1 | 5 | 0.8 | Reject now; not enough strict bit-proof for public dispatch. |

## Behavior proof obligations

Any source hunk must preserve:

- active-window stream
- shift-packet stream
- selected-`m` stream
- sweep, deflation, fallback, and exceptional-shift counters
- quasi-Schur `h` bits
- eigenvalue bits
- full-eig eigenvector bits
- complex-pair slot ordering
- RNG absence
- strict golden SHA

## Selected Pass-3 action

Attempt candidate 1 only: a same-order helper for the existing production row and
column modifications in `eig_francis_schur_traced`. It must be one hunk, with no
grouped/far-update reordering. Keep only if focused proof passes and immediate
same-worker A/B shows a real runtime win.

Reject trigger: any bit/profile drift or same-worker median win below roughly
2%. If rejected, route to a different primitive: a private grouped operation-tape
shadow replay that proves exact scalar identity before any public dispatch.
