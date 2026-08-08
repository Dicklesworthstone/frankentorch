# 27aci — the numel sweep did not discriminate: the backward stage is NON-MONOTONIC

`27aci` predicted the sweep would separate two hypotheses: linear in numel ⇒ data
movement, flat ⇒ fixed per-backward machinery. **It is neither.** Reporting that
rather than picking whichever reading is closer.

## Result

`pool_kernel_vs_tape_probe`, 15-rep medians, two runs, `avg_pool2d [N,64,64,64]` f64.
`bwd_stage` = (cumulative-through-backward − cumulative-through-sum), the same
differencing the `uufyp` decomposition uses.

| N | MiB | bwd_stage run 1 | run 2 | raw_bwd | stage/kernel | ms/MiB |
|---|---|---|---|---|---|---|
| 1 | 2 | 1.057 | 1.641 | 0.08 | 13–23x | 0.53 / 0.82 |
| 2 | 4 | 1.858 | 1.232 | 0.12 | 11–15x | 0.47 / 0.31 |
| 4 | 8 | 3.724 | 2.655 | 0.18 | 15–20x | 0.47 / 0.33 |
| **8** | **16** | **12.089** | **12.377** | 0.31–0.36 | **34–40x** | 0.76 / 0.77 |
| 16 | 32 | 6.171 | 6.360 | 4.5–5.2 | 1.2–1.4x | 0.19 / 0.20 |

## Two things this says, and one it doesn't

**1. It is not simple linear data movement.** If the stage were N full-size passes,
32 MiB would cost ~2x the 16 MiB point. It costs **half** of it — 6.2 ms against
12.1. A monotone cost model of any kind is refuted.

**2. There is a spike exactly at 16 MiB**, which is precisely the shape `uufyp`
measured and the one this whole thread has been reasoning about. Both runs agree
(12.089 / 12.377), so it is not noise: 2% apart across independent invocations while
the neighbouring points scatter by 30–40%.

**3. At 32 MiB the raw kernel itself changes regime** — 0.36 ms → 4.5–5.2 ms, a ~14x
jump for 2x the data. That is the per-CCD L3 cliff `un3os` measured independently
(this host's 128 MiB L3 is 4×32 MiB slices), showing up here from a different
direction. It is a consistency check on that earlier finding, not a new one.

**What it does not say:** where the 11–12 ms goes. The discriminant I designed does
not discriminate, so `27aci`'s question is still open.

## Why I am not explaining the spike

I can construct a story — an allocator size-class boundary, THP behaviour, the 16 MiB
buffer interacting with the L3 slice while a second live copy pushes the working set
over — and any of them would sound reasonable in a commit message. None is tested,
and this session has already produced one confident, well-argued, thoroughly measured
mechanism (`8obhh`'s buffer-size L3 predictor) that was **wrong at the level that
mattered**. A fourth plausible mechanism is worth less than the measurement that
would kill three of them.

## What to do next, in order

1. **Confirm the spike is a property of the stage and not the harness.** Run the same
   sweep with finer spacing around it (N = 6, 7, 8, 9, 10) and check the shape. A
   single-point spike at exactly the size everything else has been measured at is
   suspicious in a way that deserves ruling out before it is explained.
2. **Separate allocation from work inside the stage**, since the stage is the only
   thing left holding the 11 ms: time a backward whose gradient buffer is pre-warmed
   against one that is not, using the `condition_allocator` pattern from
   `dense_scatter_attribution` (which was itself introduced because an unconditioned
   probe measured its own lane order at 83%).
3. Only then look for a lever, and A/B it at the lane with a negative control.

## Standing

`27aci` stays **open**. `uufyp`'s finding is unaffected — `tensor_backward` really
does cost ~11 ms against a 0.32 ms kernel at the 16 MiB shape, and that remains the
largest single term on this lane. What has changed is that the simple explanations for
it are now measured to be wrong.

No PyTorch arm; nothing here is a vs-upstream claim.
