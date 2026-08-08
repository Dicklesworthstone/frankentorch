# k1h8g — avg_pool2d's gap is not in the kernels, and no kernel lever can close it

This bead has been treated as a kernel-performance problem. It is not one. FrankenTorch's
avg_pool2d **kernels are already faster than PyTorch's whole op**; everything the lane
loses, it loses outside them.

## Measured

`crates/ft-api/examples/pool_kernel_vs_tape_probe.rs`,
`executing_elf_sha256=044e002e1bd0f2bdf3685289c7038787c5511390b3758ccca8bb6edf94aa3a13`
(current HEAD, after the `8obhh` gate revert), mimalloc, 15-rep medians, three runs,
host load ~10–13.

| lane | raw_fwd | raw_bwd | kernels | session | tape overhead |
|---|---|---|---|---|---|
| `max_pool3d` | 1.355 / 0.828 / 0.828 | 1.526 / 1.638 / 1.513 | 2.881 / 2.467 / 2.342 | 7.565 / 7.866 / 6.443 | **62% / 69% / 64%** |
| `avg_pool2d` | 1.304 / 1.029 / 1.001 | 0.413 / 0.333 / 0.321 | 1.716 / 1.362 / 1.322 | 18.548 / 17.714 / 17.693 | **91% / 92% / 93%** |

`avg_pool2d`'s two kernels together are **~1.32 ms**. The same-day h2h has PyTorch's
whole avg_pool2d op at **1.701 ms** (load 86) and **2.705 ms** (load 8). So FT's
kernels are **already at or below PyTorch's entire op**, forward and backward
included.

## What this means for the bead

`k1h8g` currently reads as a kernel gap ("avg_pool2d op work ~4-6x slower"). The
current standing is 2.2–2.3x, and **none of it is available to a kernel lever**:

- Make both avg_pool2d kernels **infinitely fast** and the session arm still costs
  ~16.4 ms of the ~17.7 ms measured here.
- This is why `o5t00`'s result was not a near miss. The L3 scatter gate could not have
  helped this lane even if it had worked, because the term it targets is ~7% of the
  lane's cost.
- It is also consistent with `8obhh`: the tape is large enough to dominate, which is
  exactly the working set that made a buffer-size L3 predictor wrong.

`max_pool3d` is less extreme but points the same way — 62–69% outside the kernels.

## The caveat that bounds this, stated before anyone acts on it

**`tape_overhead` here is not purely the autograd tape.** The probe's own note says
the session arm additionally builds the leaf and sums the returned gradient, which the
raw arms do not. For this shape the input is `[8,64,64,64]` = 2M f64 = **16 MiB**, so
the session arm also pays a 16 MiB leaf construction and a 16 MiB gradient reduction
per iteration, plus a fresh `FrankenTorchSession`.

That is why this probe's session figure (17.7 ms) is much larger than the h2h's
op-work figure for the same lane (3.7–6.3 ms, leaf built outside the timer on both
sides). **The two are not the same measurement and must not be differenced.**

What survives that caveat is the ratio that both agree on: the h2h's op-work FT is
~3.7 ms against ~1.3 ms of kernels, i.e. **kernels are roughly a third of op work**,
matching this bead's own earlier note ("raw kernels are 34% of op work at median").
So the conclusion — the majority of the lane is not kernel time — holds in the
op-work framing too, even though the 91% figure is specific to this probe's framing.

## What should happen next, and what should not

**Should not:** any further kernel optimisation of `avg_pool2d_forward_f64` or
`avg_pool2d_backward_f64`. They are not the constraint, and `o5t00` already measured
one gate on the backward as a loss.

**Should:** split `tape_overhead` into its parts before choosing a lever, because
right now it is one number covering at least four different costs — session
construction, leaf materialisation, the autograd tape itself, and the gradient
reduction. A probe that times those separately would say which is worth attacking;
attacking "the tape" as a unit would be the same mistake as attacking "the dense
write" was.

Note also `project_gmuml_tape_retention` in the ledger: the session tape is recorded
as never freeing nodes, so no-grad forwards degrade linearly. If that retention also
affects this path it is a candidate explanation for a large constant per-session cost,
and it is already a known, separately-tracked issue rather than a new discovery.

No PyTorch arm in this probe; the PyTorch numbers quoted above come from the h2h runs
recorded under `frankentorch-87sz8`.
