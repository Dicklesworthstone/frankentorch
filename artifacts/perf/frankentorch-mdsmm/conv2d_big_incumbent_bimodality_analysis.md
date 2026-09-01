# conv2d_big: the standing's SIGN depends on an incumbent mode the harness does not report

`frankentorch-mdsmm`, torch:2 (CreamTrout), 2026-09-01. thinkstation1, nproc 64, rayon 8,
ELF sha256 1c7bbababe8f725b (base ba1fe627), PyTorch 2.12.1+cpu self-reported in every
invocation, guard checked before each, `concurrent_measurements=none ACTIVE` throughout.

## The six full twin invocations

| invocation | mode | FT plain | PT plain | FT masked | PT masked | plain standing | masked standing | inflation |
|---|---|---|---|---|---|---|---|---|
| run1  reps=24  2-lane | low | 4.241 | 6.333 | 8.307 | 6.566 | 1.493x | 0.790x | 1.889x |
| run2  reps=48  2-lane | HIGH | 4.473 | 10.856 | 8.702 | 11.382 | 2.427x | 1.308x | 1.856x |
| run3  reps=24  2-lane | low | 4.109 | 5.688 | 8.355 | 5.660 | 1.384x | 0.677x | 2.043x |
| run4  reps=12  2-lane | low | 4.290 | 5.730 | 8.499 | 6.100 | 1.336x | 0.718x | 1.861x |
| run5  reps=48  2-lane +slotdump | HIGH | 4.201 | 10.500 | 8.443 | 10.766 | 2.499x | 1.275x | 1.960x |
| full board  reps=16  67 lanes | low | 4.192 | 5.523 | 8.727 | 5.721 | 1.318x | 0.656x | 2.010x |

Standing > 1.000x = FrankenTorch faster. **The masked twin's standing spans
0.656x to 1.308x — it changes SIDE — and the split is exactly the incumbent's mode.**
Our arm does not move: plain 4.109-4.473 ms, masked 8.307-8.727 ms across all six.

## What IS invariant: the inflation

    inflation = [FT(masked)/FT(plain)] / [PT(masked)/PT(plain)]
    range over all six invocations, BOTH modes:  1.856 - 2.043   mean 1.937

A mode that scales the incumbent uniformly cancels in a ratio of ratios, and it does:
the inflation holds to +/-5% across a 1.85x swing in the incumbent. This is the
twin sweep's actual claim and it survives the defect below.

PyTorch pays 2.5-6.5% for the mask; we pay 95-108%.

## The defect: the incumbent arm is bimodal at ~1.85x, and the mode is fixed per invocation

22 guard-gated invocations of the same ELF on the same host, same two lanes, same env:

    conv2d_big incumbent arm     20 invocations   5.06 - 6.38 ms      (the low mode)
                                  2 invocations  10.50 - 10.86 ms     (the high mode, ~9%)
    conv2d_big OUR arm           22 invocations   4.04 -  4.47 ms      (invariant)

The mode is fixed for the whole run, not a drift into it. The per-round slot dump of a high
invocation (`FT_H2H_DUMP_SLOTS=*`, 48 rounds):

    rounds  0-11   PT median 10.471     rounds 24-35   PT median 10.469
    rounds 12-23   PT median 10.591     rounds 36-47   PT median 10.451

Flat from round 0. Whatever sets the mode is set before the first timed sample.

## Four candidate causes, all refuted by measurement

**FT_H2H_REPS.** The first five invocations split 3 low / 2 high and the split tracked REPS
exactly (12/24 low, 48 high) — which is why it was tested first. Refuted two ways. The child's
sample loop is request-driven and never learns the round count (`harness_interleave.rs:424`),
and a **round-interleaved ladder** (16/32/40/48/64, two passes, `for pass { for reps }` so host
drift cannot fake a monotone REPS effect) read the incumbent at 5.06-6.20 ms in **10 of 10**,
including four invocations at 48 and 64. The original correlation was chance; P(both highs
landing in the four REPS=48 runs) = 2.9%, and 2.9% events happen.

**Lane count.** A full board interleaves 67 lanes through one torch child instead of two, which
is a real difference in what its caches hold and is the configuration every banked row was taken
under. It reads conv2d_big PT **5.523 ms** — the low mode. (That invocation is LOAD-DRIFTED and
no ratio from it is quotable; the incumbent ABSOLUTE is what is being read here, and drift would
push it up, not down, so it argues in the same direction.)

**Host state.** Load and clocks overlap across the modes and do not separate them:

    invocation        PT       loadavg at start   in-run median core MHz
    run1  low       6.333      1.99/1.76/2.31              1429
    run2  HIGH     10.856      2.49/2.15/2.41              3416
    run3  low       5.688      3.41/2.62/2.56              2392
    run4  low       5.730      3.13/2.77/2.62              2397
    run5  HIGH     10.500      3.42/3.01/2.72              1998

The two high invocations bracket the low ones on both axes. Cross-core spread was 3.004x
(median) in all five. A six-invocation probe at fixed REPS=24 read 5.27-6.38 while loadavg
climbed 2.04 -> 7.36, so load does not drive it either.

**A peer.** `concurrent_measurements=none ACTIVE` in every invocation, and the guard passed
before each.

Cause: NOT IDENTIFIED. Stated as such rather than guessed.

## Why every gate passed anyway

Three invocations produced conv2d_big_masked rows with **all four gates PASS** — PT A/A, FT A/A,
parity `match`, drift PASS:

    run1  low    ratio 0.790 [0.735, 0.821]   1.27x SLOWER
    run2  HIGH   ratio 1.308 [1.239, 1.349]   1.31x FASTER
    run5  HIGH   ratio 1.275 [1.236, 1.297]   1.28x FASTER

Same binary, same host, same lanes, minutes apart, non-overlapping CIs, opposite signs, every
gate green. This is `feedback_aa_null_blind_to_scaled_incumbent` with a measured frequency: an
A/A null compares two positions INSIDE one run, so an incumbent scaled uniformly for the whole
run cancels out of it exactly.

## Consequences

1. **No conv2d_big standing ratio is banked from this session.** Its sign is a coin the harness
   does not report the face of.

2. **The inflation IS banked** — 1.937x mean, 1.856-2.043 over six invocations spanning both
   modes — because a uniform incumbent scaling cancels in the ratio of ratios. conv2d_big joins
   conv2d_xl (1.742x) and conv3d (1.940x).

3. **Any single-invocation vs-torch ratio on this board carries a ~9% chance of being wrong by
   1.85x**, and no gate it has can see it. The remedy is not another within-run null: it is a
   multi-invocation median of the incumbent's ABSOLUTE arm time, with a row refused when its
   incumbent deviates from that median. That is the check
   `feedback_aa_null_blind_to_scaled_incumbent` already prescribes, unautomated.

4. **The board's own banked figures for these lanes sit in the high mode** — item 203 records
   conv2d_big PT 11.019 ms and conv2d_big_masked PT 11.590 ms, against 20 of 22 invocations
   today at 5.1-6.4 ms and the full board at 5.5 ms. Either those figures were taken in the rare
   mode or the host has changed since; today's board-default configuration does not reproduce
   them. Item 203's lane SIZING argument rests on those numbers.

5. **The conv3d row banked earlier today is corroborated, not impeached.** Its incumbent read
   5.907 / 6.034 ms; today's full board reads conv3d PT 6.229 ms. Both low mode, so that row was
   not taken in the rare one. Its inflation finding (1.940x) is mode-invariant regardless.
