# conv2d_big, re-run under the incumbent-mode check

`frankentorch-mdsmm`, torch:2 (CreamTrout), 2026-09-01. thinkstation1 nproc=64 rayon=8, PyTorch
2.12.1+cpu self-reported in every invocation, guard checked before each, round_warmup=0 (board
default), FT_H2H_REPS=24 FT_H2H_WARMUP=16, lanes exact-matched.

Two ELFs, both clean-overlay: `839d9020` at base bd8fac55 (10 invocations) and `717b8cd1` at base
b82531b2 (3 invocations plus one unfiltered confirmation). Bank at
`artifacts/perf/incumbent_bank.jsonl`, committed.

## The check does what it was built for

    #  level        plain standing   masked standing
    1  NO-HISTORY        1.240x           0.647x
    2  SINGLE/n2         1.328x           0.676x
    3  LEVEL-B/2         2.822x           1.445x
    4  LEVEL-A/2         1.338x           0.691x
    5  LEVEL-A/2         1.550x           0.805x     <- CONTENDED, see below
    6  LEVEL-A/2         1.520x           0.772x
    7  LEVEL-A/2         1.447x           0.760x
    8  LEVEL-A/2         1.190x           0.629x
    9  LEVEL-B/2         2.567x           1.331x
   10  LEVEL-A/2         1.415x           0.691x
   11  LEVEL-A/2         1.371x           0.681x
   12  LEVEL-A/2         1.318x           0.653x
   13  LEVEL-A/2         1.270x           0.675x

Standing > 1 = FrankenTorch faster. **Every invocation that read the masked lane as FASTER is one
the check labelled LEVEL B**, and it labelled them at the time, in the row, without being told.

    conv2d_big_masked   level A (11 obs)   0.629 - 0.805   i.e. 1.24x - 1.59x SLOWER
                        level B  (2 obs)   1.331 - 1.445   i.e. 1.33x - 1.45x FASTER

The two levels do not overlap and the sign differs between them. That is the whole defect, now
legible in the output instead of discoverable by running the lane twenty-two times.

## What is quotable, and what is still not

Two level-A masked rows passed ALL FOUR GATES (PT A/A, FT A/A, parity, drift): ratio 0.760
[0.722, 0.800] and 0.629 [0.582, 0.669], plus an unfiltered confirmation invocation at 0.651
[0.638, 0.679] with both nulls PASS.

**Their CIs do not overlap.** The mode check separates a 1.85x bimodality; it does not make
level A homogeneous — the incumbent still ranges 5.30-6.51 ms inside it, and that ~18% moves the
ratio. So the honest standing is a RANGE conditional on the level, not a point:

    conv2d_big_masked is a standing LOSS of 1.24x - 1.59x on incumbent level A,
    and level A is where 11 of 13 invocations here, and 20 of 22 yesterday, landed.

A point estimate would be false precision. This is the first time the lane has had a standing at
all, because it is the first time a sign could be attached to a stated condition.

## The inflation replicates on a different binary

    inflation = [FT(masked)/FT(plain)] / [PT(masked)/PT(plain)]

    13 invocations, both levels:   1.880 - 2.046,  mean 1.950
      level A (n=9)                1.880 - 2.046
      level B (n=2)                1.929 - 1.953

Yesterday's figure, measured on ELF `1c7bbaba` across six invocations, was 1.856-2.043, mean
1.937. Two different binaries, two different sessions, the same number — and it is flat across a
1.85x swing in the incumbent, because a uniform incumbent scaling cancels in a ratio of ratios.
**The inflation is the durable result on this lane; the standing is the conditional one.**

## The check found something it was not built to find

Invocation 5 banked TWO records per lane at one timestamp. One process writes one record per lane
per invocation, so a collision means two harnesses measured at once — and both the pair already
banked (7.611 / 7.733 ms) and the pair that invocation measured (7.674 / 7.673) sit ~25% above
the level-A cluster its neighbours produced (5.30-6.19). **The guard had PASSED.** It checks once,
before the run, and its own header says it cannot serialise a fleet; the peer started inside that
window. `concurrent_writer_note` now reports the collision, and invocation 5 is excluded above.

Corroborated within the hour: the 12-invocation standing run was refused 9 times by the guard for
`2 peer measurement process(es) are live`. Peers are demonstrably measuring on this host, so a
collision landing inside the check-to-run window is the expected event, not a coincidence.

The bank also now flags level A as `LEVEL SPREAD EXCEEDS THE GAP — suspect a third level`, which
is the contended pair widening it. That flag is doing its job: it is pointing at observations
that should not be drawing a boundary.

## Limitations, stated

1. **It cannot classify the past.** Yesterday's 22 observations predate the check and never
   recorded the incumbent's checksum, so they cannot be backfilled. The check changes the future.
2. **A whole session at one level still reads SINGLE.** The bank gains its power by spanning
   sessions, which is why it is committed rather than left in /tmp; a bank narrower than an hour
   now says in as many words that a single-level verdict from it is weak.
3. **It separates levels, not noise.** Level A's own 18% spread is untouched, and two all-gates
   rows inside it can still disagree.
4. **Only conv2d_big and conv2d_big_masked have history.** Every other lane on the board reports
   NO HISTORY until it has been run again, and reads "one invocation short of readable" — which
   is true, and was equally true before, silently.
