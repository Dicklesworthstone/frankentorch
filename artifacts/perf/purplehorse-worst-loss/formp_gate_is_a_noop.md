# The shipped `form_p` blocking gate, measured at last — and it is a NO-OP

**Result: the `n >= 130` blocked-`form_p` dispatch produces no measurable change in the
vs-PyTorch lane ratio. 1.378x SLOWER at n=128 (unblocked) against 1.377x at n=132
(blocked).** It is neither the win it was landed as, nor the pessimization I
speculated it was one message earlier. That speculation is retracted below.

This answers the note the dispatch site has been carrying:

> UNMEASURED AS SHIPPED: the correctness case is gated, the SPEEDUP is not. Per
> section 1 of the standing orders this is landed, not won, until a post-fix ratio
> is taken in a quiet window.

It is now measured.

## The run

```
RAYON_NUM_THREADS=8 FT_GATE_SIZES="120,124,128,132,136,140" \
FT_GATE_VALUES="262144,262144" \
PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of target/release/examples/bidiag_gate_sweep_h2h>
```

* `elf_sha256=323a90cf1804eed18f1f4f62ae8ec4e04357403b4c697981755cd144ecfb8848`,
  self-reported by the running process.
* Incumbent **PyTorch 2.12.1+cpu**, threads=8, driven as a co-process **inside the
  same invocation** and self-reporting its own version.
* `FT_GATE_VALUES=262144,262144` is the shipped arm **twice**, so arm1-vs-arm0 is a
  real A/A null rather than an arm compared with itself.
* Unpinned. 9 rounds, first discarded, arm order reversed on odd rounds, every
  ratio the median of the paired per-round ratio.
* Window: idle **90.36% then 88.42%** immediately before launch (mpstat over 5 s,
  twice), 0 iowait. `n=120..140` is 115-157 KB of f64 and stays cache-resident.

## The rows

| n | form_p route | PT min | FT min | vs PyTorch | A/A nulls | form_p/q | reduction |
|---|---|---|---|---|---|---|---|
| 120 | unblocked | 1.224 ms | 1.682 ms | 1.328x SLOWER | 1.000, 0.998 | 0.296 ms | 0.731 ms |
| 124 | unblocked | 1.139 ms | 1.325 ms | 1.278x SLOWER | 1.000, 0.998 | 0.229 ms | 0.607 ms |
| 128 | unblocked | 1.281 ms | 1.862 ms | 1.378x SLOWER | 1.000, 0.997 | 0.336 ms | 0.766 ms |
| 132 | **BLOCKED** | 1.334 ms | 1.665 ms | 1.377x SLOWER | 1.000, 0.998 | 0.446 ms | 0.794 ms |
| 136 | **BLOCKED** | 1.397 ms | 1.657 ms | 1.220x SLOWER | 1.000, 1.013 | 0.448 ms | 0.978 ms |
| 140 | **BLOCKED** | 1.599 ms | 1.944 ms | 1.312x SLOWER | 1.000, 1.002 | 0.453 ms | 0.895 ms |

Every A/A null is inside ±0.015 of 1.0. Parity MATCH at every size (rel 2.1e-13 to
4.4e-13). Every incumbent figure passed the plausibility gate against
pre-session banked values.

**Across the gate the lane ratio does not move**: 1.378x at n=128 and 1.377x at
n=132, inside a spread (1.22–1.38x) that the six sizes show anyway.

## The retraction, and why the number that fooled me is not evidence

One message earlier I read the `form_p/q` phase counter jumping from ~0.29 ms below
the gate to ~0.45 ms above it, backed `form_q` out on an equal-flop assumption, and
said blocked `form_p` looked "~2x slower than unblocked" and the gate "mis-set".
**That was wrong, and the sweep above is what shows it.**

The phase counter really does step at the gate — 0.296/0.229/0.336 below,
0.446/0.448/0.453 above, and the blocked cluster is strikingly tight. But:

* **The step is invisible in the lane.** `form_p/q` gains +0.162 ms across the gate
  while the FT lane minimum *falls*, 1.862 ms at n=128 to 1.665 ms at n=132.
  Phase totals rise while the thing they are phases of gets faster.
* **The two figures are different estimators and must not be differenced.** The
  phases are a *median of 3 instrumented calls*; the lane is a *min over 9 rounds*.
  Subtracting one from the other is the error my own notes forbid — a min and a
  median of the same work have read 1.512x apart on this host.
* **The phase counter was already under suspicion.** Item 258c found the same
  counters summing to 1058 ms against a 464 ms measured median and wrote: "No
  phase percentage should be quoted from a single call until that is resolved."
  That caveat applies here and I quoted through it.
* **The unblocked cluster is not even monotonic** — n=124 (0.229 ms) reads *below*
  n=120 (0.296 ms), which is unphysical. An instrument that inverts on a 1.10x size
  step cannot resolve a 1.5x claim.

The honest statement is the lane one: **the gate is a no-op at the threshold**, and
the phase counter's step is an artefact of the instrument, not a property of the
code.

## What this closes and what it leaves open

**Closed.** Raising, lowering or removing the `n >= 130` gate is *not* a lever. It
buys nothing at the threshold and there is no pessimization to undo. Anyone tempted
by the "UNMEASURED AS SHIPPED" comment can stop here.

**Still open, and now with one leg knocked out.**
`frankentorch-bidiag-form-q-unblocked-gl0rj` observes that `form_p` has a blocked
compact-WY path while `form_q` has none, despite identical ~2n³/3 flops. That
*source* asymmetry is unchanged and still real. But this sweep removes the
measurement I had hoped would size it: since the `form_p` gate is a no-op, the
discontinuity cannot be used to separate `form_p` from `form_q` inside
`SVD_FORM_PQ_NS`, and the phase counter is not trustworthy enough to do it either.

Sizing that lever now needs one of:

* a **values-only vs full SVD pair** at one size — `tensor_linalg_svdvals` runs with
  `track_left = false` and therefore never calls `form_q`, so the difference is
  `form_q` directly, measured at the lane level with the same estimator on both
  sides rather than through the phase counters;
* or splitting `SVD_FORM_PQ_NS` into two counters, which is a sentinel edit to
  `ft-kernel-cpu` and is being avoided while that crate is mid-publish.

The first needs no code change to the kernel and is the next measurement.

## Standing

Unchanged and independent of all of the above: the SVD square forward is a **LOSS**
against PyTorch at every size measured this session — 1.22–1.38x at n=120–140 here,
and 1.748x at n=256 / 2.515x at n=512 in commit `75d3fad3`. Parity is not the
price: MATCH at 1e-13 throughout.
