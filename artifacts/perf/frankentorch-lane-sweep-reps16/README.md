# REPS-16 re-bank of the four gauntlet loss lanes — a fresh set, on two oracles

## What this is, and what it replaces

The four loss-lane figures in circulation — `max_pool3d` 7.31x, `avg_pool2d`
6.00x, `conv3d` 3.49x, `max_pool1d` 2.24x — were each **a single run at
`REPS=15`**, on the rep count `frankentorch-svabf` later proved
position-biased, and **no executing-ELF digest was recorded**. Direction and
rough size were safe to carry; the digits were not.

This artifact banks a **fresh set** on the fixed `REPS=16` harness with its ELF
recorded, across **36 runs on two PyTorch oracles**. It is **not** differenced
against the old digits.

## Read this before any number below: the arms are not interleaved

The harness runs its **entire PyTorch arm to completion before the first
FrankenTorch lane starts** (`gauntlet_lane_sweep_h2h.rs:196` waits out the Python
child; the first FT lane is at `:264`). So every ratio here is
**same-invocation and same-host, but NOT interleaved** — the two arms are
sampled tens of seconds apart, and any load shift in that gap lands entirely and
undetectably in the ratio.

**What makes these numbers tolerable is 18 repetitions and a median — not the
contention preflight.** A preflight certifies only that nothing heavy sat on the
placement CPUs at the instant sampling began; it cannot see a peer job starting
one second later, and it cannot see page-cache or thermal history at all. On this
host that is not hypothetical: a neighbouring project's oracle cycled between
idle and 4000–6900% CPU throughout these runs, and load average moved between 6
and 69.

Two direct consequences:

- **A single run of this harness is not a measurement**, no matter how clean its
  preflight looked or how convincingly its A/A gate passed.
- **The residual risk is one-sided and unquantified.** Repetition averages the
  gap's effect down; it does not eliminate it, and nothing here bounds it. Treat
  the ranges as the honest width of each lane, not the medians as exact.

Interleaving the arms per repetition would remove this defect outright and is the
single highest-value change available to this harness. It is deliberately **not**
made here, so this artifact's ELF stays the one that produced these numbers.

## Provenance

| | |
|---|---|
| harness | `crates/ft-api/examples/gauntlet_lane_sweep_h2h.rs` (`REPS=16`) |
| executing ELF SHA-256 | `7286dcfc85bc6c77caff8b434be4429f05a4261e75fd011f1b0dc70d54fb982c` |
| self-reported by the process | yes — printed from inside the run, identical across all 36 runs |
| allocator | mimalloc (`--features fair-alloc`), self-reported |
| measurement | OP WORK ONLY — forward+backward, leaf built outside the timer on **both** sides |
| host | 64 cores, governor `performance`, shared with a live agent swarm |
| PyTorch arm | live, in the same invocation, min-of-7 after 4 warmups, `torch.set_num_threads(8)` |
| arm ordering | **NOT interleaved** — the whole PT arm completes before the first FT lane; see the section above |
| **oracle B (primary)** | **torch 2.12.1+cpu** — `/data/projects/.venvs/frankentorch-pytorch-cpu`, the repo's standing oracle |
| oracle A (control) | torch 2.13.0+cpu |
| runs | 18 per oracle, same ELF throughout |
| gradient parity | **36/36 `match`** (FT grad sum vs PyTorch, 1e-6 relative) |

## Banked set — oracle B, torch 2.12.1 (quote these)

| lane | FT ms (med) | PT ms (med) | **ratio median** | range | ratio spread | A/A gates | parity |
|---|---|---|---|---|---|---|---|
| `max_pool3d` | 5.309 | 0.718 | **7.45x SLOWER** | 5.85–8.53x | 1.46x | 17/18 PASS | 18/18 |
| `avg_pool2d` | 8.011 | 1.149 | **6.87x SLOWER** | 3.09–8.07x | 2.61x | 18/18 PASS | 18/18 |
| `conv3d` | 21.049 | 5.530 | **3.77x SLOWER** | 3.19–4.42x | 1.39x | 17/18 PASS | 18/18 |
| `max_pool1d` | 17.413 | 6.976 | **2.43x SLOWER** | 1.14–3.18x | 2.79x | 18/18 PASS | 18/18 |

**Bank the median with its range. A single run of this harness is not a
measurement** — that is exactly what produced the digits being replaced.

`max_pool3d` remains the largest confirmed vs-PyTorch loss in the tree, and it
and `conv3d` are the two *decidable* lanes (ratio spread 1.46x and 1.39x).

## The old digits were not wrong — they were measured on this oracle

| lane | old single-run digit | oracle B median [range] | verdict |
|---|---|---|---|
| `max_pool3d` | 7.31x | **7.45x** [5.85–8.53] | reproduces |
| `avg_pool2d` | 6.00x | **6.87x** [3.09–8.07] | reproduces |
| `conv3d` | 3.49x | **3.77x** [3.19–4.42] | reproduces (old at low edge) |
| `max_pool1d` | 2.24x | **2.43x** [1.14–3.18] | reproduces |

All four old figures land inside the fresh range on the canonical oracle. The
REPS-15 position bias and the missing ELF made them **uncertifiable**, not
wrong. They are now certified, with digits and a digest.

## The finding: two of the four lanes move with the PyTorch version

Running both oracles on the identical ELF isolates the incumbent as a variable,
and it is not a small one.

| lane | ratio on B (torch 2.12.1) | ratio on A (torch 2.13.0) | PT ms B → A | version effect |
|---|---|---|---|---|
| `max_pool3d` | 7.45x | 7.78x | 0.718 → 0.658 | **robust** (1.04x) |
| `conv3d` | 3.77x | 3.89x | 5.530 → 5.420 | **robust** (1.03x) |
| `avg_pool2d` | 6.87x | 4.21x | 1.149 → 2.089 | **1.63x** — PT got 1.82x slower |
| `max_pool1d` | 2.43x | 1.29x | 6.976 → 13.502 | **1.88x** — PT got 1.93x slower |

FrankenTorch's own arm barely moved between the two sets (FT medians 17.413 vs
17.129, 8.011 vs 7.780, 5.309 vs 5.145, 21.049 vs 20.756 — all within 3%). **The
entire effect is in the incumbent.** PyTorch 2.13.0 is roughly 1.8–1.9x slower
than 2.12.1 on `max_pool1d` and `avg_pool2d` on this host.

The consequence is a trap worth naming: **upgrading the oracle would have
"improved" `max_pool1d` from 2.43x to 1.29x and `avg_pool2d` from 6.87x to 4.21x
with no FrankenTorch change whatsoever.** A claim is only as good as the arm it
was measured against — including that arm's *version*, which belongs in
provenance alongside host, thread count, and ELF.

## The gate defect: A/A PASS does not mean the host was quiet

One oracle-A run measured `max_pool3d` at **29.22x** — FT spiked to 19.186 ms
against a normal PT arm of 0.657 ms. **That run's A/A gate PASSED**, null CI
`[0.528, 1.359]`.

The mechanism is structural. The A/A null compares FrankenTorch against
FrankenTorch, so any disturbance scaling both arms cancels exactly; what
contention actually does is **widen the null CI**, and a wider CI brackets 1.0
more easily. The clean run's CI for that lane was `[0.798, 1.266]` (width
0.468); the 29.22x run's was `[0.528, 1.359]` (width 0.831) — **78% wider, and
still a PASS.**

> **The A/A null gate gets *easier* to pass as the host gets noisier.** It is
> anti-conservative in precisely the condition it is trusted to detect.

The two observed FAILs (both on oracle B under load average 47–68) confirm the
gate is not inert, but also how arbitrary the bracketing criterion is: `conv3d`
failed on `[1.011, 4.453]` — a CI of width 3.44, wild by any standard, that
failed only because its lower bound cleared 1.0 by 0.011.

Two consequences the campaign should adopt:

1. **A passing A/A gate certifies that the two FT arms were treated alike. It
   certifies nothing about the PyTorch arm and nothing about host quiet** — by
   construction, PyTorch is not in the null.
2. **CI width, not just bracketing, is the quiet-host signal.** A null that is
   centred *and tight* is evidence of a calm sample; merely centred is not.

Recording this as an observed defect class rather than fixing it in the same
change: a gate edit must be its own change with its own before/after integrity
check, and this artifact's ELF must stay the one that produced these numbers.

## Why `avg_pool2d` and `max_pool1d` are the wide lanes

Their FT arms are stable (FT spread 1.18x and 1.10x on oracle B — the *tightest*
arms in the set). Their PyTorch arms are bimodal. Oracle-B per-run PT for
`max_pool1d`, in run order:

```
14.694  5.292  5.861  7.232  6.285  6.436  5.728  6.291 14.818
 5.702 13.931 11.648 11.125  9.852  5.482 14.354  6.879  7.072
```

A ~5.5 ms mode and a ~14 ms mode, roughly 2.6x apart, in the incumbent. That
bimodality is the entire source of the 1.14–3.18x ratio spread. **The movement
is in PyTorch, not in us** — `frankentorch-k1h8g` suspected this from two runs;
36 same-ELF runs establish it.

## What is quotable

- **Quotable:** the oracle-B medians with their ranges, above, naming torch 2.12.1.
- **Quotable:** `max_pool3d` (7.45x) and `conv3d` (3.77x) are decidable and version-robust.
- **Quotable:** all four prior digits reproduce on the canonical oracle.
- **NOT quotable:** any single-run point estimate from this or any prior sweep.
- **NOT quotable:** `avg_pool2d` or `max_pool1d` without naming the torch version.
- **NOT quotable:** any comparison between a ratio here and a ratio from a different run.

## Reproducing

```
cargo build --release -p ft-api --features fair-alloc --example gauntlet_lane_sweep_h2h
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
  ./target/release/examples/gauntlet_lane_sweep_h2h
```

Run the built binary **directly**; it must execute locally because the rch
workers have no PyTorch, and the harness hard-fails rather than emitting an
FT-only number if the PyTorch arm did not run. **Run it at least 10 times and
take the median**, and record which torch version was the incumbent.
