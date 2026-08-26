# n=1024 banked at last — 3.05x SLOWER, and it is BETTER than the number it replaces

**Result: the SVD square forward at n=1024 is 3.046x SLOWER than PyTorch — measured
in the quietest window of the session (loadavg 6.3), incumbent verified plausible,
parity MATCH. This is the worst standing in the tree, and it is the row that was VOID
on the first attempt.**

It also *corrects downward* the figure the campaign has been carrying: item 257b read
**3.94x / 3.43x** at this size; this reads **3.05x / 2.95x**.

## The run

```
RAYON_NUM_THREADS=8 FT_GATE_SIZES="512,1024" FT_GATE_VALUES="262144,262144" \
PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of target/release/examples/bidiag_gate_sweep_h2h>
```

* `elf_sha256=323a90cf1804eed18f1f4f62ae8ec4e04357403b4c697981755cd144ecfb8848`
  — **deliberately the same ELF as commit `75d3fad3`**, so this row slots directly
  beside that one's n=256 (1.748x) and n=512 (2.515x) with no cross-binary caveat.
* Incumbent **PyTorch 2.12.1+cpu, threads=8**, co-process, same invocation.
* `FT_GATE_VALUES=262144,262144` — the shipped arm **twice**, so arm1-vs-arm0 is a
  genuine A/A null.
* Window: idle **88.80% then 88.97%** (mpstat 5 s, twice) immediately before launch.
  **loadavg 6.27 → 7.63** during the run — the quietest this host has been all
  session, against the 78–175 it spent most of it at.
* Incumbent plausibility, against pre-session banked figures: n=512 PT min
  **30.037 ms** (banked ~32.075) and n=1024 PT min **124.083 ms** (banked ~122.604) —
  both **ok**.

## The rows

| n | FT min (arm0 / arm1) | PT min | standing | **A/A null** | PT spread | parity |
|---|---|---|---|---|---|---|
| 512 | 68.277 / 67.300 ms | 30.037 ms | **2.401x / 2.396x SLOWER** | **1.015 — PASS** | 1.26x | rel 1.90e-12 MATCH |
| 1024 | 356.682 / 325.046 ms | 124.083 ms | **3.046x / 2.946x SLOWER** | 1.039 | 1.14x | rel 6.07e-14 MATCH |

**n=512 carries a passing A/A null (1.015)** and replicates commit `75d3fad3`'s
2.515x on the same ELF — the standing is stable across windows.

**n=1024's null is 1.039**, missing the ±0.02 band by 1.9%. It is quoted as measured,
not certified. The two arms read 3.046x and 2.946x — a 3.4% spread, consistent with
that null and far too small to threaten the conclusion that this size is ~3x.

## What it replaces

The first attempt at this row (commit `75d3fad3`, kept in `svd_run1.log`) was **void**
and was discarded rather than banked: PyTorch's own spread hit **952x**, loadavg went
12.6 → 125 mid-row as six peer benches started, iowait hit 4282 jiffies, and the two
arms disagreed by 1.4x (5.304x against 3.789x) where at n=256/512 they agreed within
4%. Discarding it was correct — it would have been the largest number in this campaign
and it was an artefact.

This row is what that one was supposed to be: same ELF, same configuration, a window
an order of magnitude quieter, and the two arms now agreeing to 3.4%.

## And it corrects the campaign's figure downward

Item 257b banked n=1024 at **3.94x / 3.43x**. That row's own provenance notes
`iowait 367` at n=1024 specifically, against 0–51 at every smaller size — i.e. the
largest size was the one its window handled worst. This run reads **3.046x / 2.946x**
at iowait 110 and loadavg 6.3.

Both numbers are ours; the newer one comes from the quieter window and the arms agree
more tightly, so **~3.05x is the figure to carry**. The direction of the correction is
worth stating plainly: the standing was *too harsh* on us by roughly 25%, and it is
still the worst loss in the tree.

## The board after this

| standing | ratio | note |
|---|---|---|
| **SVD square forward n=1024** | **3.05x SLOWER** | worst in tree; null 1.039, measured not certified |
| `conv2d_f32_masked` | 2.78x SLOWER ±4% | HEAD's fix already took it from 4.49x |
| SVD square forward n=512 | **2.40x SLOWER** | **null 1.015 PASS** |
| SVD square forward n=256 | 1.75x SLOWER | |
| SVD n=120–140 | 1.22–1.38x SLOWER | |
| `conv2d_f32` | 1.02x — parity | |

The expansion phase (`form_p + form_q`) is 33.5% of the n=512 lane and removing all of
it still leaves 1.734x (commit `c4d611c4`), so it cannot close this on its own. The
`x86-64-v3` flag is worth ~1.07x at n=512 and nothing at n=256 (commit `636ce5bc`), so
it cannot either.
