# `slogdet` factorised the same matrix TWICE per no-grad forward — 1.93-1.98x, predicted before it was written

**Result: removing one of two identical LU factorisations moves `slogdet` from 21.3-21.6x
to 11.244x / 10.798x at n=1024 (1.93-1.98x faster) and from 24.9-26.4x to 14.715x /
14.684x at n=512 (1.70-1.79x). All four A/A nulls in band. Parity residuals unchanged to
every printed digit. Gate: `cargo test --release -j2 -p frankentorch-api --lib det` ->
29 passed, 0 failed.**

Pre-fix ELF `13355d7df8e14debd748b703dc7aa3051872496b11e3a2dc6c5cfe8c2d4e659b`
Post-fix ELF `d1843b95d3e227ae416333f0be6d8ce6ae5c0f2484869e03b1ad4f3d2fbd65ba`

| n | before | after | speedup | A/A null (after) | parity before/after |
|---|---|---|---|---|---|
| 512 | 24.933-26.444x | **14.715 / 14.684x** | 1.70-1.79x | 1.000 / 0.998 | 3.28e-13 / 3.28e-13 |
| 1024 | 21.314-21.626x | **11.244 / 10.798x** | 1.93-1.98x | 1.000 / 1.007 | 3.75e-13 / 3.75e-13 |

## The defect

`tensor_linalg_slogdet` called `slogdet_contiguous_f64` **twice per forward on the same
matrix**: once to take `.sign` (discarding `logabsdet`), then again inside the
`tensor_apply_function_with_create_graph` forward closure to take `.logabsdet` (discarding
`sign`). One result struct carries both fields. We ran the full LU twice and threw half of
each away.

The second call is legitimate when a tape node is needed — the closure must be replayable
under `create_graph`, which is the DAC contract. **With no grad there is no closure to
replay**, so both outputs come off one call. The fix is a no-grad fast path, bit-exact by
construction: same kernel, same input, both fields from one result.

## The prediction

Source reading produced three confident wrong answers earlier in this campaign, so the
mechanism was committed as a falsifiable prediction BEFORE the fix was written:

> a no-grad fast path drops slogdet n=1024 from 93-106 ms to ~50-55 ms, i.e. 21.3x ->
> ~10-11x. If it barely moves, this reading is wrong and I will say so.

Measured: **10.798-11.244x**. The ratio prediction landed; the absolute time came in better
than predicted (39.6-43.3 ms against a predicted 50-55 ms).

## Why the cross-run A/B is admissible here

This compares two runs of two different ELFs, and cross-run comparison is normally invalid
on this host — the incumbent has moved 1.94x between two runs of the SAME ELF, and an A/A
null is blind to a uniformly scaled incumbent because it compares two positions inside one
run.

**The incumbent did not move**: torch's own absolute time is 4.950 ms before vs 4.520 ms
after at n=1024 (1.10x), and 1.333/1.073 vs 1.205 ms at n=512. Both FT figures are also
paired per-round against a live in-invocation torch, so each ratio is internally valid;
what is being compared across runs is two ratios, not two raw timings.

## How it was found — an internal inconsistency, not a torch comparison

`inv` was added to test whether the map row "slogdet/LU — 21.3-24.9x" named a cause or just
bundled an op with a route. It refuted the row: `inv` does strictly MORE work
(getrf + an O(n^3) tail vs getrf + an O(n) log-product) and landed 3.2-3.9x better.

That exposed the real signal, which needs no reference implementation to read:

> at n=1024 our `slogdet` (93.4-106.0 ms) was SLOWER than our own `inv` (79.1-82.5 ms)
> while doing strictly less work — where torch has them the sensible way round
> (4.95 ms vs 15.94 ms).

Same shape as the Householder proof ("our own `qr` is 32.9x faster than our own
`geqrf`+`orgqr`"). Post-fix the ordering is restored: slogdet 39.6-43.3 ms < inv 79.1-82.5 ms.

## The vein is ONE OP DEEP, not a family

Checked every sibling with the same signature — a kernel producing a non-differentiable
side-output that the autograd wrapper cannot return:

| op | side-output | no-grad path | verdict |
|---|---|---|---|
| `cummax`/`cummin` | indices | moves both fields out of one call | clean |
| `det` | none | 2nd call is in the BACKWARD closure | clean |
| `lu_factor` | pivots | short-circuits to `tensor_variable(result.lu)` | clean |
| **`slogdet`** | **sign** | **still ran the create_graph closure** | **defect** |

`slogdet` was the only op whose no-grad path went through the autograd closure anyway, so
it paid for a replayability it never used. `lu_factor`'s grad-path second factorisation is
inside a replayable closure and is the DAC contract's price, not waste — it is correct as
designed and was NOT changed.

## What is NOT fixed

The **grad path still factorises twice**, and that is not safely removable by the same
means: the closure must be replayable, so capturing a precomputed result would break replay
semantics. Not attempted.

## Residual, now cleanly isolated

Post-fix `slogdet` is getrf plus an O(n) tail, so its remaining **10.8-11.2x is
approximately our bare getrf gap** against torch's — the largest non-Householder
single-matrix gap at n=1024, above `eigh`'s 8.2x.

One inference deliberately NOT drawn: decomposing `inv` into getrf + getri by subtracting
slogdet's time. Invalid — `inv` computes `A^-1 = solve(A, I)` through the batched LU-solve
kernel and never runs classical getri, so the two ops do not share a getrf implementation.
