# un3os step 3 — the sibling call sites, each measured on its own

`lever_in_situ_ab.md` gated ONE call site and said explicitly that the siblings
were left alone because "widening an effect measured at one site is how zoqws went
wrong". This measures them. Four candidates; **three win, one does not, and the one
that does not is left ungated.**

## Method

One ELF pair of `crates/ft-api/examples/dense_scatter_attribution.rs` with a lane
per kernel. Arm A flips `dense_scatter_should_parallelize` to always-true, which
disables **every** gate at once; arm B is the shipped gate. Each lane still calls
only its own kernel, so one pair yields independent per-kernel rows.

| arm | ELF sha256 |
|---|---|
| A — all gates off | `58a1ce5167ca57cc…` |
| B — gated | `ecf70ac87a3f8f76…` |

9 paired rounds, arms alternating with the lead arm flipping. Shapes chosen to sit
**below** the 16 MiB gate, since that is the side the gate changes.

## The A/A control moved, and correcting for it changes a conclusion

The two byte-identical `alloc_only` lanes came back at **0.705 and 0.746** — not
1.0. Those lanes call no gated kernel, so a raw reading of this table would have
been wrong.

The cause is a whole-process effect specific to this comparison: arm A runs *five*
kernels in parallel where arm B runs them serially, so arm A leaves 64 rayon threads
hot and every arm-A lane — including the controls — pays for it. **Arm A is inflated
~38% across the board.** (The single-kernel A/B in `lever_in_situ_ab.md` did not have
this problem — its controls were flat at 0.958–1.044 — because its arms differed in
only one kernel. The two results are consistent once that is accounted for.)

So every ratio below is **normalised by the per-round control factor**.

## Result

| kernel | raw B/A | **normalised** | speedup | 95% CI (norm) | verdict |
|---|---|---|---|---|---|
| `max_pool3d_backward_from_indices_f64` | 0.350 | **0.472** | 2.12x | [0.260, 0.573] | WIN (already shipped) |
| `max_pool3d_backward_from_indices_scalar_f64` | 0.297 | **0.447** | **2.24x** | [0.233, 0.469] | **WIN → gated** |
| `max_pool2d_backward_from_indices_f64` | 0.364 | **0.557** | **1.80x** | [0.312, 0.619] | **WIN → gated** |
| `max_pool1d_backward_from_indices_f64` | 0.478 | **0.706** | **1.42x** | [0.423, 0.806] | **WIN → gated** |
| `max_pool3d_backward_2x2s2_f64` | 0.638 | **0.975** | 1.03x | **[0.731, 1.122]** | **inconclusive → NOT gated** |

## The negative result is the useful one

`max_pool3d_backward_2x2s2_f64` shows **no effect this harness can resolve** — its
CI straddles 1.0. Its gate was written, measured, and then **reverted**.

That is not noise, it is mechanism. The gate helps when the pass does almost no work
per byte, so memory behaviour dominates. The four winners each write one scattered
value per output element. The 2x2s2 kernel instead recomputes an **8-way max over
the input plane** for every output element — far more work per byte — so the memory
term does not dominate and there is nothing for serialisation to recover.

This also **bounds the vein**: it is not "pooling backward", it is
*trivial-work scatter into an L3-resident buffer*. Sites that compute anything
substantial per element are outside it. That prediction is worth testing before the
next site is gated, rather than assuming the gate generalises.

## Cross-check against the earlier isolated measurement

`kernel_scatter` appears in both A/Bs and is the consistency check:

- isolated single-kernel A/B, flat controls: **4.24x**
- this five-kernel A/B, control-normalised: **2.12x**

The isolated figure is the better estimate *for that kernel* (its arms differed in
one thing, and its controls confirmed it). The normalised 2.12x here also lands on
top of the attribution probe's original ~2x prediction, which is reassuring: the two
harnesses disagree in a way that is fully explained by how many kernels each arm
changed, not by the kernel behaving differently.

## Scope

Still not touched: the **avg_pool** backward family. Those accumulate over a
`kh×kw` window per output rather than writing one value, so by the mechanism above
they are *expected* to behave like 2x2s2 rather than like the winners. Untested
either way — no gate, and no claim.

No PyTorch arm anywhere in this file; nothing here is a vs-upstream claim.
