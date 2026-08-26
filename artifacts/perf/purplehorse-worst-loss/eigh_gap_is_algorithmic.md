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

## Bounding lever 1: divide-and-conquer alone CANNOT close this

The `eigh − eigvalsh` split does not need torch, and `linalg_gap_sweep` already
carries both halves in one invocation. Absolute times below are heavily
contention-inflated (this host was at **0.04% idle**; eigh n=512 reads 222 ms here
against 64.3 ms in the certified clean window), so **only the within-invocation ratios
are used** — and even those are suspect in a knowable direction, since contention hurts
the parallel replay more than the serial-scalar reduction and therefore *inflates* the
vector share.

| threads | n | eigh | eigvalsh | vectors | share |
|---|---|---|---|---|---|
| 8 (contended) | 256 | 39.20 | 17.97 | 21.23 | 54.2% |
| 8 (contended) | 512 | 222.30 | 76.45 | 145.85 | 65.6% |
| 8 (contended) | 1024 | 1292.38 | 570.18 | 722.20 | 55.9% |
| 64 (lighter load) | 512 | 78.33 | 42.27 | 36.06 | **46.0%** |

So the eigenvector path is **roughly half to two-thirds** of `eigh`, and the share is
strongly thread- and load-dependent — the replay is `par_chunks_mut` parallel while the
reduction is documented "serial scalar", so more threads shrink the replay's share.

**That materially qualifies the ledger's 73%.** It was taken at 10 threads; nothing
here reproduces it, and the honest range is 46–66%. The *direction* stands — vectors
dominate — the magnitude does not.

### The bound

Applying the range to the certified **5.599–5.628x** at n=512, and assuming
divide-and-conquer made the eigenvector phase **entirely free** (it would not; it makes
it cheaper):

| vector share | a FREE eigenvector phase leaves |
|---|---|
| 46% (low) | **3.02x SLOWER** |
| 66% (high) | **1.90x SLOWER** |

**Lever 1 alone cannot close `eigh`.** Even in the impossible best case it lands
between 1.9x and 3.0x slower — i.e. roughly where the SVD square forward already sits.
The reduction half must be attacked too, which is lever 2 (`dsytrd` blocked
tridiagonalisation), the one the source calls "the real lever … a multi-turn rewrite".

This is the same discipline that bounded the SVD expansion phase at 1.734x even if
wholly removed (`c4d611c4`), and it lands the same way: **the single obvious lever is
worth having and is not sufficient.** Anyone scoping the D&C rewrite should scope it as
"5.6x → ~3x", not as "closes eigh".

## Correcting my own framing: the 9x flop ratio is real but says little about TIME

I opened this file with "the replay does nine times the reduction's arithmetic" and
hedged that a flop ratio is not a time ratio. The hedge was right and the reason is now
quantitative, and it inverts the emphasis:

| n | reduce (SERIAL) | back-transform (SERIAL) | replay (PARALLEL) | serial total, 1 core | replay per core, 8 threads |
|---|---|---|---|---|---|
| 512 | 0.179e9 | 0.089e9 | 1.61e9 | **0.268e9** | **0.201e9** |
| 1024 | 1.432e9 | 0.716e9 | 12.88e9 | **2.147e9** | **1.611e9** |

**Per core the two halves are comparable** — 0.268e9 against 0.201e9 at n=512. That is
why the measured split is ~50/50 and not the 9:1 the raw flop count suggests, and it
means I was pointing at the wrong half. The replay has nine times the arithmetic and
spreads it over every thread; the reduction and back-transform have far less and run
**entirely on one core**.

`eigh_tred2_backtransform` is fully serial and nothing in the file marks it as a
target. At n=1024 that is **2.1e9 flops on a single core of a 64-core machine.**

## A bounded, bit-exact lever the "it regressed" note does not cover

The source's PERF NOTE says rayon-parallelising the reduction "was MEASURED and
REGRESSED (eigh 256 77->85ms): at the benched **n<=256** the per-step work (~i²) is too
small to amortize the fan-out". That negative result is scoped to n≤256. At n=512 the
per-step work is 4x larger and at n=1024 16x — the same "validated at one shape" trap
that made the conv3d direct-kernel gate a 1.5–3.3x pessimisation above its test shape.

Inside `eigh_tred2_backtransform` the two inner loops are **not** equally constrained:

```rust
for k in 0..i {                                   // PROJECTION — reduction over k
    let row_factor = row_i[k];
    for j in 0..i { projections[j] += row_factor * row[j]; }
}
for k in 0..i {                                   // UPDATE — independent over k
    let reflector = previous_rows[k * n + i];
    let row = &mut previous_rows[k * n..k * n + i];
    for j in 0..i { row[j] -= projections[j] * reflector; }
}
```

* The **projection** loop accumulates into `projections[j]` across `k`. Parallelising it
  reassociates a reduction and changes bits — the same wall that keeps FMA out of the
  SVD row-dot. Closed.
* The **update** loop is embarrassingly parallel over `k` and **bit-exact**: row `k`
  owns the disjoint slice `[k*n, k*n+i)`, `projections` is read-only, and the read at
  `previous_rows[k*n + i]` sits outside every mutated slice. No element's arithmetic
  changes and nothing is reassociated. `par_chunks_mut` over rows is legal here.

That is half of 2n³/3 currently on one core, recoverable bit-exactly.

**NOT implemented, deliberately.** The host has sat at 0.01–0.12% idle throughout, so a
perf claim cannot be validated right now, and this session's standing rule is that an
unmeasured change is "landed, not won". The bit-exactness argument is provable and the
existing eigh tests would catch an error, but shipping a perf edit I cannot measure —
at the end of a budget, into a crate a peer is mid-publish on, with agent-mail's DB
corrupt so no reservation can be recorded — is the wrong trade. It is written down
here in full so it can be picked up and measured rather than rediscovered.

**Size it before building it**: the serial pair is ~50% of `eigh` per the split table,
the update loop is half of the back-transform, so the ceiling is roughly
`0.089e9 / 0.268e9 ≈ 33%` of the serial half ≈ **~17% of eigh** — worth ~1.2x, not a
gap-closer. Consistent with everything else here: real, bounded, insufficient alone.
