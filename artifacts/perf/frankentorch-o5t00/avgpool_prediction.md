# o5t00 — the avg_pool prediction, tested. Both of my hypotheses were wrong.

`un3os` bounded the L3-residency serial gate and left a falsifiable prediction:
avg_pool backward "should behave like `2x2s2` (no resolvable effect), because each
output accumulates over a `kh×kw` window rather than writing one value."

That prediction is **refuted**, and not in the direction I hedged toward either.

## What I predicted, twice, and got wrong twice

1. **In the bead:** avg_pool would show *no effect*, like `2x2s2`.
2. **On re-reading the kernels before measuring:** I doubted my own bead. `2x2s2`'s
   extra cost is *re-reading the input plane* to recompute an 8-way max, and
   avg_pool backward reads no second array at all — it divides and writes. So I
   expected it might behave like the *winners*.

Neither. Gating avg_pool backward is an **active LOSS**.

## Result

Same harness as `sibling_ab.md`: one ELF pair, a lane per kernel, arm A flips
`dense_scatter_should_parallelize` to always-true, 9 paired rounds, arms
alternating, all ratios normalised by the per-round A/A control factor (0.842 here).

| kernel | A med | B med | raw | **normalised** | 95% CI (norm) | verdict |
|---|---|---|---|---|---|---|
| `max_pool3d_..._from_indices` (anchor) | 0.897 | 0.392 | 0.433 | **0.511** → 1.96x | [0.469, 0.559] | WIN |
| `avg_pool2d_backward_f64` | 0.761 | 0.871 | 1.086 | **1.266** → 0.79x | [1.108, 1.502] | **LOSS** |
| `avg_pool1d_backward_f64` | 1.040 | 1.198 | 1.137 | **1.289** → 0.78x | [1.185, 1.646] | **LOSS** |

Both avg_pool CIs sit **entirely above 1.0** — this is not an inconclusive result
like `2x2s2`, it is a measured ~1.27x slowdown. The anchor reconfirming 1.96x in the
same run is what rules out a bad window.

Both gates were written, measured, and **reverted**. DO-NOT-GATE notes carrying
these numbers now sit at both call sites.

## The boundary, corrected

`un3os` called the vein "trivial-work scatter into an L3-resident buffer". The
work-per-byte half of that was the wrong discriminator. The right one is
**sparsity of the write**:

- The max_pool scatters write **one value per output element** — 1-in-8 of the
  buffer. They touch every cache line while doing almost no work, and a single core
  keeps the whole thing L3-resident. Serialising wins.
- avg_pool backward spreads each output across its whole `kh×kw` window, so it
  writes **every element**, with arithmetic attached. That is enough real work to pay
  for the threads. Serialising throws that away.
- `2x2s2` sits between: sparse writes, but heavy input re-reads. No resolvable
  effect either way.

So the vein is **SPARSE scatter**, not "dense write" and not "trivial work". That is
a narrower claim than `un3os` made, and it is the one supported by data.

## Why this was worth running even though it changed nothing shipped

Three sites are now protected by measured DO-NOT-GATE notes (`2x2s2`,
`avg_pool2d_backward_f64`, `avg_pool1d_backward_f64`) instead of by nobody having
tried. Without this, the natural next move for any agent reading `un3os` — "same
`vec![0.0;n]` + `par_chunks_mut` shape, apply the same gate" — would have shipped a
1.27x regression across the avg_pool family, and it would have looked principled.

## Scope

Only `avg_pool2d_backward_f64` and `avg_pool1d_backward_f64` were measured. The
scalar and `2x2s2` avg_pool variants were not, but they share the dense-write shape,
so the same DO-NOT-GATE conclusion is the expected one — **expected, not measured**,
and they carry no note. No PyTorch arm anywhere here; nothing is a vs-upstream claim.
