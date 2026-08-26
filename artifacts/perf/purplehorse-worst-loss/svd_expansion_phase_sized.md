# The SVD expansion phase, sized at lane level — 33.5% of n=512, and removing ALL of it still leaves a loss

**Result: `form_p + form_q` is 1.504x of the square SVD at n=512 — i.e. 33.5% of the
lane — measured with one estimator on both sides. And the bound that matters:
deleting the entire expansion phase would move the standing from 2.466x SLOWER to
1.734x SLOWER. It would not close the gap.**

That bound is the useful output here. It says `frankentorch-bidiag-form-q-unblocked-gl0rj`
is worth at most ~1.42x on this lane and cannot be the whole answer, which is worth
knowing before anyone writes a blocked `form_q`.

## Why a new instrument was needed

`SVD_FORM_PQ_NS` wraps `form_p` and `form_q` in one counter and cannot size either.
Worse, the counter is not trustworthy in absolute terms:

* item 258c caught it summing to 1058 ms against a 464 ms measured median;
* my own n=120..140 sweep caught it **non-monotonic** — 0.229 ms at n=124 against
  0.296 ms at n=120, which is unphysical;
* it is a *median of 3 instrumented calls* while the lane is a *min over 9 rounds*,
  and differencing those two estimators is an error this campaign has already paid
  for (a min and a median of the same work have read 1.512x apart on this host).

`svd_blocked_bidiag_values` — the values-only prologue — **materialises no
reflectors**, so `tensor_linalg_svdvals` skips `form_p` and `form_q` entirely.
`full − values` is therefore the whole expansion phase at the **lane** level, same
estimator on both sides. `bidiag_gate_sweep_h2h.rs` gained `FT_VALUES_ARM=1` to
carry it as a third arm, appended last so every existing arm keeps its index.

**It is an arm, not a second invocation**, deliberately: a cross-run subtraction on
this host is worthless — the incumbent has moved 1.94x between two runs of the same
ELF. Both halves interleave round-by-round inside one process against one live
PyTorch.

## The run

```
RAYON_NUM_THREADS=8 FT_GATE_SIZES="256,512" FT_GATE_VALUES="262144,262144" \
FT_VALUES_ARM=1 PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of target/release/examples/bidiag_gate_sweep_h2h>
```

* `elf_sha256=5650da0f49b04c2a91dad92759b09667ab7fac057c60e1642170c33aa4233d64`
* Incumbent **PyTorch 2.12.1+cpu, threads=8**, co-process, same invocation.
* `RAYON_NUM_THREADS=8` matched to the harness's `torch.set_num_threads(8)`.
* Window: idle >= 70% over 5 s, twice, immediately before launch. Incumbent
  plausibility checked against pre-session banked figures: n=512 PT min 27.748 ms
  against ~32.075 ms banked — **ok**.
* arms[0] and arms[1] are the shipped config **twice** (a real A/A null); arms[2] is
  the same config values-only.

## n=512 — quotable

| arm | min | vs PyTorch | paired-vs-arm0 | parity |
|---|---|---|---|---|
| [0] full SVD | 71.976 ms | **2.466x SLOWER** | 1.000x | rel 1.90e-12 MATCH |
| [1] full SVD (A/A null) | 80.100 ms | 2.433x SLOWER | **0.985x — PASS** | rel 1.90e-12 MATCH |
| [2] **VALUES-ONLY** | 50.092 ms | **1.734x SLOWER** | **1.504x** | rel 1.90e-12 MATCH |

PT min 27.748 ms, spread 1.33x, iowait 303 jiffies, MHz 4197/4286, load 17.94→18.19.

**The expansion phase is 1.504x**, the harness's own paired per-round median ratio —
not a subtraction of two mins by me. That is **33.5% of the full lane**, against the
28% the phase counter claimed. The counter understated it.

### The parity column is load-bearing here, not decoration

`svdvals` and `svd` must return the **same singular values**, so the checksum
agreeing (`rel 1.90e-12 MATCH` on all three arms, identical across them) is a live
check that the subtraction compares two forms of the *same* decomposition rather
than a truncated one. A MISMATCH would have made the whole figure meaningless.

## n=256 — NOT quotable

| arm | min | vs PyTorch | paired-vs-arm0 |
|---|---|---|---|
| [0] full SVD | 12.079 ms | 1.689x SLOWER | 1.000x |
| [1] full SVD (A/A null) | 11.732 ms | 1.754x SLOWER | **0.928x — FAIL** |
| [2] VALUES-ONLY | 7.520 ms | 1.140x SLOWER | 1.577x |

The A/A null misses by 7.2% against a ±2% band and **PyTorch's own spread was
4.55x** in this row. Two identical arms disagreeing by 7% means the window could not
resolve the 1.577x it reports. Recorded, not quoted. The n=512 row is the one that
carries the finding.

## What this says about where the gap actually is

The campaign's standing belief is that the reduction — 70-74% of our time — is *the*
target, and items 253-255 were pointed at it. This reframes that:

* Our **values-only** SVD at n=512 (50.092 ms) is already **1.734x slower than
  PyTorch's ENTIRE SVD** (27.748 ms, U and S and Vh included). The reduction and QR
  sweep alone lose to torch doing strictly more work.
* The expansion phase adds a further 21.9 ms, taking 1.734x to 2.466x.

So **both halves are losses**, and neither alone explains the gap. The reduction is
the bigger absolute term; the expansion is the better *ratio* lever because it is
33.5% of the lane and `form_q` within it is still BLAS-2 while `form_p` is BLAS-3.

**But the ceiling is now known and it is not enough.** A perfect blocked `form_q`
cannot do better than removing the whole expansion phase, which lands at 1.734x —
still a loss. Anyone scoping that work should scope it as "2.47x → ~2.0x", not as
"closes the SVD gap".

## Consequences for the open bead

`frankentorch-bidiag-form-q-unblocked-gl0rj` keeps its source finding — `form_p` has
a blocked compact-WY path, `form_q` has none, despite identical ~2n³/3 flops — and
now has a measured size and a measured ceiling. What it still does **not** have is
the split *between* `form_p` and `form_q` inside that 33.5%. The `n >= 130` gate
turned out to be a no-op (commit `7ebc0555`), so the dispatch discontinuity cannot
separate them, and this instrument measures the pair. Splitting them needs a second
counter in `ft-kernel-cpu` — deferred while that crate is mid-publish.

The equal-flop argument predicts roughly a 50/50 split, which would put `form_q` at
~17% of the lane and a blocked `form_q` at ~1.2x on the standing. That is a
prediction, not a measurement, and is flagged as such.
