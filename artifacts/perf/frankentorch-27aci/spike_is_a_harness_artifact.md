# 27aci — the spike is a HARNESS ARTIFACT, and it contaminates uufyp's headline number

Step 1 of `27aci` was "confirm the spike is a property of the stage and not the
harness". It is the harness. **The spike moves when you reverse the sweep order.**

## The test

Same probe, finer spacing, run **both directions in one invocation**. If the cost is a
property of the shape it must sit at the same N in both passes.

| N | MiB | ascending | descending |
|---|---|---|---|
| 5 | 10 | 2.210 | 3.277 |
| 6 | 12 | 4.063 | 1.983 |
| 7 | 14 | 3.909 | 3.523 |
| **8** | 16 | **11.895** | **1.820** |
| **9** | 18 | **13.893** | **2.379** |
| 10 | 20 | 13.979 | 13.996 |
| 12 | 24 | 16.658 | 17.335 |

**N=8 costs 11.9 ms ascending and 1.8 ms descending — a 6.5x swing on identical work.**
N=9 swings 13.9 → 2.4.

## What actually predicts the cost

Not the size. **Whether the size is at or above the largest one allocated so far in
the process.**

- Ascending, every N is a new high-water mark from N=8 up, and every one of those is
  expensive; the small ones that ran first are cheap.
- Descending, N=12 and N=10 are the new high-water marks and are expensive; everything
  after them — including the same N=8 and N=9 that were expensive ascending — is cheap,
  because a larger block has already been faulted and can be recycled.
- N=10 and N=12 are expensive in **both** passes, which is the control: they are the
  largest sizes either way, so they never get to reuse a bigger block.

That is allocator/first-touch behaviour, not engine work. It is the *same* confound
that made two byte-identical lanes read **83% apart** in
`dense_scatter_attribution.rs`, and which `condition_allocator()` was written to fix —
**and this sweep has no conditioning at all.**

## The uncomfortable consequence: uufyp's 11 ms is suspect

`uufyp` (commit 81c9d4ad) reported `tensor_backward` at ~11 ms against a 0.32 ms
kernel, ~34x, and concluded the next lever belongs in `ft-autograd`. That
decomposition **also has no allocator conditioning**, and its backward lane is the
first in its cumulative chain to allocate the 16 MiB *gradient* buffer.

Given the numbers above, an unknown and possibly dominant share of that 11 ms is
first-touch cost for a not-yet-seen allocation size, not autograd engine work.

**I am not claiming uufyp is wrong.** The backward stage may still be the largest term
once conditioned — the forward lane allocates 16 MiB too, so some warming already
happens, and the descending pass still shows real spread. What I am saying is that
**the 34x figure is not currently defensible**, and I put it in a commit message and a
bead comment as though it were.

## What has to happen before anything is built on it

1. **Add `condition_allocator()` to the decomposition lanes and re-run.** The pattern
   already exists in `dense_scatter_attribution.rs`; it dirties and frees a same-sized
   block outside the timed region so every lane faces the same precondition.
2. **Add an A/A pair at different positions in the cumulative chain**, which is what
   exposed the 83% ordering effect in the first place. The decomposition currently has
   no null lane at all — that is the gap that let this through.
3. Only then re-state where the session time goes.

## Standing

- `27aci` step 1: **done, negative** — the spike is not real.
- `uufyp`: **reopened in effect.** Its conclusion ("the lever belongs in ft-autograd,
  not ft-kernel-cpu") is unproven, not disproven. Its *other* findings are unaffected
  and were separately controlled: `session_new` is free, and `leaf_build` ≈ a bare
  clone — both are comparisons against a same-size baseline rather than absolute
  first-touch costs.
- `k1h8g`'s core claim survives independently: avg_pool2d's **kernels** are ~1.3 ms
  against PyTorch's ~1.7–2.7 ms whole op. That comes from the raw kernel lanes, which
  do not depend on the session decomposition at all.

## The lesson, again, one level up

I built `condition_allocator` in this session precisely because an unconditioned probe
measures allocator history. Then I wrote two new probes without it and drew a
conclusion from one of them. **The defence has to be applied to every new harness, not
remembered as a story about an old one.**
