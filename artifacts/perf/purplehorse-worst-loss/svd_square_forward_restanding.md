# The worst vs-PyTorch loss, re-derived and re-measured — SVD square forward

**Result: a LOSS, confirmed. FT is 1.75x slower at n=256 and 2.52x slower at n=512
than PyTorch 2.12.1+cpu.** The n=1024 row from the same invocation is VOID and is
not quoted; see [n=1024 is void](#n1024-is-void-and-why-that-is-not-a-detail).

## What the worst loss actually is

The prompt named the masked route (~2.58x) as the likely worst. It is not. Ranking
every standing in the tree by magnitude:

| standing | ratio | provenance |
|---|---|---|
| **SVD square forward, n=1024** | 3.43–3.94x SLOWER | item 257b — best instrument on record |
| `conv2d_f32_masked` | 4.55x SLOWER *claimed* | **a doc comment only** — see below |
| SVD square forward, n=512 | 2.38–2.39x | item 257b |
| `conv2d_big_masked` | 2.42–2.44x | banked, 3 certified rows |
| SVD square forward, n=256 | 1.82–1.85x | item 257b |

Two things worth stating before any of that is acted on:

* **The 4.55x `conv2d_f32_masked` figure has no ledger entry.** It appears only as a
  doc comment on `conv2d_backward_dinput_direct_f32` in `ft-kernel-cpu/src/lib.rs`,
  attributed to `frankentorch-hi9r6`. `grep` over the whole of
  `docs/NEGATIVE_EVIDENCE.md` (2.5 MB) and all of `artifacts/` finds it nowhere
  else — no null, no window, no ELF hash. It is a claim, not a banked row.
* **HEAD landed an unmeasured fix for exactly that lane.** `6c7aef5f` (2026-08-25)
  added `conv2d_backward_dinput_direct_f32` *and* its dispatch. Nobody has priced
  it. That measurement is queued separately and is not part of this document.

So the worst *confirmed* loss is the SVD square forward, and it is what this
re-measures.

## The measurement

One invocation. PyTorch is driven as a **co-process inside that same invocation**
and self-reports its own version, so the incumbent cannot be a figure from another
window. `RAYON_NUM_THREADS=8` is matched to the harness's own
`torch.set_num_threads(8)`, which is the configuration item 257b used and the only
one these numbers are comparable with.

```
RAYON_NUM_THREADS=8 \
FT_GATE_SIZES="256,512,1024" \
PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of target/release/examples/bidiag_gate_sweep_h2h>
```

* `elf_sha256=323a90cf1804eed18f1f4f62ae8ec4e04357403b4c697981755cd144ecfb8848`
  — **self-reported by the running process**, not computed from a file afterwards.
* Host `thinkstation1`, 64 cores. Binary snapshotted to scratchpad before the run
  and the snapshot timed, so a peer rebuilding the shared target dir mid-run
  cannot change what was measured.
* Incumbent: **PyTorch 2.12.1+cpu**, threads=8, both self-reported in the same
  invocation. The version is pinned deliberately — the same ELF has read 2.43x
  against one torch build and 1.29x against another.
* `scripts/measurement_window_guard.sh` **PASSED** before the run
  (`guard_clear=yes after 6 checks` — the window cleared on its own after 3 min).
* 9 rounds, first discarded, arm order reversed on odd rounds; every ratio is the
  median of the **paired per-round** ratio, with the incumbent sampled once per
  arm rather than in a block.

### The rows

| n | FT min | FT median | PT min | PT spread | standing | parity |
|---|---|---|---|---|---|---|
| 256 | 11.572 ms | 13.364 ms | 6.785 ms | 1.30x | **1.748x SLOWER** | rel 3.09e-13 MATCH |
| 512 | 73.993 ms | 84.065 ms | 32.075 ms | 1.14x | **2.515x SLOWER** | rel 1.90e-12 MATCH |
| 1024 | — | — | — | **952.35x** | **VOID** | — |

Load was 11.53 → 12.61 across the n=256 and n=512 rows, CPU 3.09–3.43 GHz.

These replicate item 257b (1.85x at n=256, 2.39x at n=512) within the window's
noise, in a separate session on a fresh binary. **The standing holds.**

### Parity is not the problem

`rel 3.09e-13` and `1.90e-12` on the singular-value sum, MATCH at every size. We
are not buying our slowness with accuracy, and there is no tolerance question here
to hide behind.

## n=1024 is void, and why that is not a detail

The n=1024 row printed `5.304x SLOWER` on one arm and `3.789x` on the other. It is
not quotable, and the run says so in its own output:

* **PyTorch's own spread was 952.35x** across its samples. An incumbent that varies
  by three orders of magnitude within one row is not measuring anything.
* **loadavg went 12.61 → 124.99 during the row.** Six peer measurement processes
  started while it ran (a criterion sweep, a frankenfs bench, two frankenpandas
  harnesses, a pytest).
* iowait 4282 jiffies, against 51 and 305 on the two rows that are good.
* The guard, re-run after the sweep, **refused on both counts**: peer processes
  live, and loadavg 124.99 over its ceiling of 35.
* The two arms disagreed by 1.4x (5.304x vs 3.789x) where at n=256 and n=512 they
  agreed within 4%. That disagreement is the window, not the code.

Quoting `5.30x SLOWER` here would have been the largest number in this document and
it would have been an artefact. It is recorded and discarded.

## What this run does NOT establish

**It carries no A/A null.** The harness default `FT_GATE_VALUES=262144,u64::MAX`
runs the shipped gate and the always-serial arm — two *different* configurations.
The `paired-vs-arm0 1.000x` printed against arm0 is 1.000 by construction (arm0
compared with itself) and is **not** a null. The harness header says exactly this:
*"null: repeat an arm in FT_GATE_VALUES — two identical arms differ only by this
window's noise."* A run with `FT_GATE_VALUES=262144,262144` is owed before either
row is called certified rather than measured.

## Where the time goes, and the target nobody has taken

Phase split, ours only, median of 3 instrumented calls:

| n | reduction | form_p/q | QR sweep |
|---|---|---|---|
| 256 | 8.024 ms (74%) | 2.642 ms (24%) | 0.211 ms (2%) |
| 512 | 48.193 ms (71%) | 18.993 ms (28%) | 0.294 ms (0%) |

The reduction is the named target and has had items 253–255 pointed at it.
**`form_p/q` is 24–28% and has had nothing pointed at it.** At n=512 our expansion
phase alone (18.993 ms) is 59% of PyTorch's *entire* SVD (32.075 ms). Item 258c
measured the same share at n=1024 (28%) and called it "a second target and nothing
has been pointed at it"; that is still true, and it is now confirmed at two more
sizes.

Note the phase percentages here come from the same instrument item 258c flagged as
untrustworthy on the serial arm (its counters summed to 2.3x the measured wall
time). On the arms above the totals are self-consistent with the medians, but the
caveat travels with the numbers.

## Provenance of the void row, kept deliberately

Raw output for all three sizes, including the discarded one, is in
`svd_run1.log` alongside this file. A void row that is thrown away silently is
indistinguishable from a row that was never taken.
