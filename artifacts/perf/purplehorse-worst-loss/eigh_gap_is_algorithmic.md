# The `eigh` 5.6x is ALGORITHMIC — we run QL iteration where LAPACK runs divide-and-conquer

**Result, from flop counting and source reading rather than a stopwatch: our
eigenvector path accumulates O(n³) Givens rotations whose flop count is ~9x the
tridiagonal reduction's, at both n=512 and n=1024. LAPACK's `dsyevd` — which is what
`torch.linalg.eigh` calls — uses divide-and-conquer (`dstedc`) and does not pay that.
No divide-and-conquer implementation exists anywhere in this tree. That is the shape
of the certified 5.599–5.628x loss, and it is why the bandwidth fix already shipped on
this phase did not close it.**

## The arithmetic

Our eigenvector path is `eigh_tql2_z_deferred` → `eigh_tql2_collect_ops` (the QL
recurrence, logging the ordered rotation stream) → `eigh_tql2_replay_blocked` (replay
across all `z` rows). The replay's inner operation is a 2×2 Givens rotation, ~6 flops,
applied for every logged rotation × every row:

| n | rotations logged (~2n²) | rotations replayed (×n rows) | replay flops | reduction flops (4n³/3) | **ratio** |
|---|---|---|---|---|---|
| 512 | 524,288 | 268,435,456 | **1.61e9** | 0.18e9 | **9.0x** |
| 1024 | 2,097,152 | 2,147,483,648 | **12.88e9** | 1.43e9 | **9.0x** |

The replay does **nine times the reduction's arithmetic**, by construction, at both
sizes. This is a property of the algorithm, not of a window, a thread count, or an
instrument.

It is also independently consistent with the ledger's single-matrix stage profile at
n=1024 — reduce 444 ms / form-Q 180 ms / **tql2-replay 1698 ms = 73%** — which I had
flagged as coming from an untrustworthy class of phase counter. The flop count reaches
the same conclusion by a route that cannot be confounded by contention, so the 73% can
now be believed in direction even if not in precise magnitude.

## The algorithm mismatch, confirmed in source

* Ours is the **EISPACK lineage**: `tred2` (Householder tridiagonalisation, doc says
  "EISPACK `tred2` lineage. O(4/3 n³)") followed by **`tql2`** — the doc reads
  "Implicit-shift QL eigenvector iteration with DEFERRED whole-stream replay."
* `torch.linalg.eigh` dispatches to LAPACK **`?syevd`**, which is `dsytrd` +
  **`dstedc`** — divide-and-conquer. `dstedc` obtains eigenvectors by merging
  sub-problems with rank-one updates and does **not** accumulate a rotation stream
  across all rows.
* `grep -c "divide_and_conquer|dc_eig|stedc"` over `ft-kernel-cpu/src/lib.rs`
  returns **0**. There is no D&C tridiagonal eigensolver in this tree.

The source already concedes the consequence, at line 27681:

> PERF NOTE: serial scalar, ~half of eigh's cost (the other half is tql2). eigh is
> FAST … but **~11x slower than LAPACK syevd**.

## Why the fix that already shipped did not close it

BlackThrush's row-blocked Givens replay (`76993cd1`) is worth **2.31x at n=512 and
3.59x at n=1024**, found by identifying the replay as *bandwidth*-bound: the per-row
`par_chunks_mut(n)` form re-streams the whole `ops` Vec once per row, ≈n× the Vec in
RAM traffic, and blocking 8 rows reads it once per block instead. `block ≥ 16` falls
off a cache cliff.

That was a real win and it is in the shipped binary. **It optimised the constant factor
of an asymptotically worse algorithm.** The certified 5.6x is what remains *after* it,
which is exactly what one expects when the residual gap is algorithmic rather than
bandwidth: you can make an O(n³) accumulation stream memory efficiently and still lose
to something that never does the accumulation.

This is worth stating plainly because the obvious next move — another bandwidth pass on
the same structure — is the one thing this evidence argues against.

## The two levers, both unimplemented, and their sizes

1. **Divide-and-conquer tridiagonal eigensolver (`dstedc` equivalent).** Replaces the
   O(n³) rotation accumulation. By the flop table this is the larger half — 9x the
   reduction's arithmetic. Nothing exists; this is a from-scratch implementation.
2. **Blocked tridiagonalisation (`dsytrd`).** The source names it directly: *"The real
   lever is BLOCKED tridiagonalization (LAPACK dsytrd: panel + symmetric rank-2k
   trailing update via `gemm::dgemm`), the same BLAS-3 family that won
   blocked-cholesky/QR — a multi-turn rewrite."* This addresses the reduction half,
   i.e. the smaller ~1/9.

Both are pure-Rust and neither needs a C BLAS/LAPACK. Lever 1 is where the flops are.

## What is still owed, and not assumed

The lane-level confirmation is queued and has not run — the host has been at 0.02–0.12%
idle throughout. `eigvalsh` skips the eigenvector path entirely
(`eigh_tql2_values_only` exists and is separate), so `eigh − eigvalsh` prices the
replay with one estimator on both sides, the same subtraction that sized the SVD
expansion phase and replaced a counter I did not trust (`c4d611c4`).

**The flop count says what the algorithm costs, not what it takes.** A 9x flop ratio
does not automatically mean a 9x time ratio — the replay is bandwidth-shaped and the
reduction is serial-scalar, so their achieved rates differ. The measurement remains
owed and the prediction stands on the record: if the 73% holds, `eigvalsh` at n=512
should land near ~17 ms and be a far smaller loss than eigh's 5.6x. If `eigvalsh` is
*also* ~5x slower, the replay is not the story and lever 2 becomes primary.

## Standing

`eigh` n=512: **5.599–5.628x SLOWER**, A/A null **1.011 PASS**, parity 6.12e-16 MATCH,
`elf_sha256=9e98e2eb1f7676c41a5eb40c13f8e05baeceaffbde75aca6a4c92e4c0eede73e`
(commit `1c571aa8`). Bead
`frankentorch-eigh-single-matrix-worst-loss-vb95f`.
