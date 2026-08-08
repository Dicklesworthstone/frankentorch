# 87sz8 — the deciding two-ELF lane A/B: ran it, and it is INCONCLUSIVE

I specified this run three times as the thing that would settle whether `un3os`'s
kernel wins reach the lane. I built it and ran it. **It does not settle it**, and the
reason is worth more than a forced verdict would have been.

## The design (this part is sound and reusable)

Two FT ELFs of `gauntlet_lane_sweep_h2h` differing **only** in
`dense_scatter_should_parallelize`:

| arm | ELF |
|---|---|
| A — gates OFF (pre-`un3os` behaviour) | `af66bcc72183b1cb…` |
| B — gated (shipped) | `93e8de27a87f5efa…` |

5 rounds, arms alternating with the lead arm flipping, same torch 2.12.1 arm.

**The metric is the per-round ratio-of-ratios** `(FT/PT)_B / (FT/PT)_A`, which
cancels both the torch arm and the host state. **And it carries its own control:**
the PyTorch arm is byte-identical in both ELFs, so `pt_B/pt_A` **must** sit near 1.0.
If it doesn't, that round is host drift and cannot be used.

## Why the result is not usable

**1. The A/A gate discarded most rounds.** Both arms must PASS in the same round for
that round to be paired:

| lane | rows | PASS | A passed | B passed | **both** |
|---|---|---|---|---|---|
| `max_pool3d` | 10 | 5 | 3 | 2 | **1** |
| `max_pool1d` | 10 | 8 | 4 | 4 | **3** |
| `avg_pool2d` | 10 | 4 | 3 | 1 | **1** |
| `conv3d` | 10 | 10 | 5 | 5 | **5** |

The two lanes I actually care about survived with **n=1**. A bootstrap CI over one
sample is degenerate — it prints a tight interval that means nothing.

**2. The control is violated on exactly those lanes.** `|pt_B/pt_A − 1|` per paired
round:

```
max_pool3d   7%
max_pool1d   11%  2%  8%
avg_pool2d   34%          <- the torch arm "moved" 34% between two identical binaries
conv3d       21%  1%  9%  2%  11%
```

A 34% swing in an arm that is byte-identical in both ELFs is pure host noise. Load
climbed from ~10 to ~34 across the run.

**So the numbers the analysis printed — "max_pool3d 1.507, GATE HURTS LANE" — are an
artifact of n=1 with a 7% control violation, and I am not reporting them as a
finding.** Had I quoted that table, it would have looked like a clean refutation of
my own session's work, complete with a confidence interval, and it would have been
worthless.

## What the run does establish

- **The harness is right, the sample is too small.** Nothing about the design needs
  changing; it needs more rounds in a quieter window.
- **A concrete acceptance rule for next time**, which this run lacked: require
  `|pt_B/pt_A − 1| < 0.05` per round *in addition to* both A/A gates passing, and
  require **n ≥ 8 surviving rounds** before computing anything. On this run that rule
  would have admitted **zero** rounds for `max_pool3d` and correctly refused to
  produce a verdict, instead of producing a confident-looking wrong one.
- **Rough cost to do it properly:** ~4 min per round-pair, and roughly half of rounds
  are discarded at load ≥ 10, so **n ≥ 8 surviving needs ~16 round-pairs ≈ 1 hour of
  genuinely quiet host.** That is the honest price of this answer, and it is why the
  question has stayed open.

## Standing

`87sz8` remains open. The question — do the four gated max_pool backward kernels
(individually 1.4–2.2x faster, bit-exact, flat controls) move the op-level lane —
remains **unanswered**. Two whole-lane readings in
`h2h_remeasure_20260808.md` showed no sign that they do; this A/B was meant to
decide it and could not.

Nothing about the shipped gates changes on the strength of this run. They are
individually measured wins with their own controls; this is a statement about the
lane, not about the kernels.
