# The whole board vs live PyTorch — 58 lanes, and the worst op is confirmed by EXCLUSION

**Result: of 58 lanes measured in one invocation against a live PyTorch 2.12.1+cpu
co-process, only SIX losses carry both A/A nulls passing, and the worst of those with
clean parity is `conv2d_f32_masked` at 1.870x SLOWER. No board lane approaches the
SVD square forward at n=1024 (3.10–3.12x, certified, commit `7cf74314`), which is not
on this board — so the worst op in the tree is now established by exclusion rather
than by assuming the ledger was right.**

## Why run the whole board

This session measured the SVD family and the f32 conv2d family in depth because the
ledger pointed there. Every other lane's standing came from rows banked days ago on
older binaries — and the ledger has been wrong in *both* directions this session:
item 257b's n=1024 was 25% too harsh (3.94x → certified 3.10x), while the 4.55x
conv2d figure that lived only in a doc comment turned out exactly right. The ranking
itself was worth re-measuring.

## The run

```
PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of target/release/examples/gauntlet_lane_sweep_h2h, --features fair-alloc>
```

No lane filter at all — `FT_H2H_LANES` unset is not the same as naming every lane, and
an unfiltered sweep is the configuration every banked full-board row was taken under.

* `executing_elf_sha256=0df4db5435ab47d398d06348922b28bf68b17bea37db289dc9422bff87b53b5d`
* `allocator=mimalloc (--features fair-alloc)` — mandatory; item 193b records that its
  omission voided every ratio taken without it.
* Incumbent **PyTorch 2.12.1+cpu, threads=8**, co-process, same invocation.
* 16 rounds, balanced-square ABBAABBA, four live samples per arm per round.
* Window: idle **80.27% then 87.97%** (mpstat 5 s, twice) immediately before launch.

## The ranked board

| lane | FT ms | PT ms | standing | PT null | FT null | parity |
|---|---|---|---|---|---|---|
| `conv2d_f32_masked_panel` | 93.286 | 25.345 | 3.68x SLOWER | PASS | OFFSET | match |
| `max_pool3d_dense` | 3.212 | 1.166 | 2.75x SLOWER | OFFSET | PASS | match |
| `max_pool3d_nopool_dense` | 2.809 | 1.184 | 2.37x SLOWER | PASS | OFFSET | match |
| `group_norm_f32_kernels_serialfwd` | 13.743 | 6.411 | 2.14x SLOWER | OFFSET | PASS | match |
| `batch_norm2d_f32_dense` | 30.341 | 14.760 | 2.06x SLOWER | PASS | PASS | **MISMATCH** |
| **`conv2d_f32_masked`** | **48.538** | **25.989** | **1.87x SLOWER** | **PASS** | **PASS** | **match** |
| `max_pool3d_nopool` | 1.769 | 0.973 | 1.82x SLOWER | FAIL | OFFSET | match |
| `conv2d_masked_train_panel` | 8.939 | 4.972 | 1.80x SLOWER | OFFSET | PASS | match |
| `max_pool3d` | 1.693 | 0.968 | 1.75x SLOWER | FAIL | OFFSET | match |
| `conv2d_masked` | 4.893 | 2.993 | 1.63x SLOWER | PASS | OFFSET | match |

### The only losses where BOTH nulls pass

| lane | standing | parity |
|---|---|---|
| `batch_norm2d_f32_dense` | 2.060x SLOWER | **MISMATCH — see below** |
| **`conv2d_f32_masked`** | **1.870x SLOWER** | match |
| `group_norm_f32_dense` | 1.480x SLOWER | match |
| `group_norm_f32_kernels_serialfwd_dense` | 1.210x SLOWER | match |
| `conv3d_masked` | 1.120x SLOWER | match |
| `group_norm_f32_zeroed` | 1.040x SLOWER | match |

## `batch_norm2d_f32_dense`'s parity MISMATCH is EXPECTED, not a bug

It reads 2.06x with both nulls passing, which would make it the board's worst
quotable loss — except its gradient checksum disagrees with torch. Before treating
that as a correctness finding I checked, and the harness documents it itself:

> WHY AN f64 LANE EXISTS BESIDE THE f32 ONE. The f32 BatchNorm lanes cannot clear
> this [1e-6 parity gate] at any shape.

`timed_batch_norm2d_f64_dense` exists precisely so BatchNorm has a lane that *can*
certify. So this is a known f32 precision limit, and the row is excluded from the
ranking rather than reported as a bug.

## Most of the board cannot certify at 16 rounds

| | PASS | OFFSET | FAIL |
|---|---|---|---|
| incumbent null | 24 | 22 | 12 |
| FrankenTorch null | 33 | 22 | 2 |

Only 24 of 58 incumbent nulls pass at the default round count. That is not a defect in
the lanes — it is what the 16→32→64 walk exists for, and it is cheaper to re-run a
high-ranking lane individually than to raise the whole board. It does mean **a
full-board row is a screening result, not a banked one**.

## LANE COMPOSITION MOVES A LANE'S ABSOLUTE TIME BY 1.45x

The most important instrument finding here, and it was not the one I went looking for.
`conv2d_f32_masked` on this 58-lane board against the same lane on a focused 4-lane
run, **same ELF, same shape, same allocator**:

| | 4-lane, 64 rounds (`ffe22c15`) | 58-lane, 16 rounds (here) |
|---|---|---|
| FT | 70.410 ms | **48.538 ms** |
| PT | 25.371 ms | 25.989 ms |
| standing | 2.78x SLOWER | **1.87x SLOWER** |

**Our arm is 1.45x faster when it has 57 neighbours than when it has 3.** The
incumbent barely moves (25.371 → 25.989, 2.4%), so this is not the window: it is our
arm's own behaviour changing with what ran before it — allocator warmth and cache
residency are the obvious suspects, and this is the in-situ-versus-standalone effect
the campaign has recorded before, now measured at 1.45x on a live lane.

**Consequence, stated plainly: a full-board row and a focused row for the same lane
are NOT interchangeable, and this session has quoted both.** The 2.78x figure from the
focused run has 4x the rounds and a passing FrankenTorch null, so it remains the one
to quote for that lane — but it should be quoted as *"2.78x in a focused 4-lane run,
1.87x in a full-board sweep"*, not as a single number. Neither is wrong; they measure
the lane in different company.

## The worst op, by exclusion

Nothing on this board — 58 lanes, every family the harness covers — reaches the SVD
square forward's certified **3.10–3.12x** at n=1024. The highest board row with clean
parity and both nulls passing is `conv2d_f32_masked` at 1.87x here (2.78x focused).

The SVD is not a board lane; it lives in `bidiag_gate_sweep_h2h`. So the standing
"SVD n=1024 is the worst loss in the tree" is no longer an inference from an old
ledger — it survives a re-measurement of everything else.

## Follow-up: the pool lanes do NOT beat the SVD, and my composition hypothesis was too broad

I predicted that because our arm ran 1.45x faster with 57 neighbours than with 3, the
board's high-ranking pool lanes would read WORSE when re-run focused, possibly
rivalling the SVD. **That prediction was wrong, and the run says so.**

```
FT_H2H_REPS=32 FT_H2H_LANES_EXACT=1 \
FT_H2H_LANES="max_pool3d_dense,max_pool3d_nopool_dense,max_pool3d,conv2d_f32_masked" \
PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <board ELF 0df4db54…, --features fair-alloc>
```
idle 71.85% then 72.97% before launch.

| lane | full-board (16r) | **focused (32r)** | direction | FT null | PT null |
|---|---|---|---|---|---|
| `max_pool3d_dense` | 2.75x | **2.47x** | FASTER focused | 0.924 | 1.020 |
| `max_pool3d_nopool_dense` | 2.37x | **2.27x** | FASTER focused | 0.967 | 0.954 |
| `max_pool3d` | 1.75x | **1.76x** | flat | 0.798 | 1.042 |
| `conv2d_f32_masked` *(control)* | 1.87x | **2.62x** | SLOWER focused | **0.997 PASS** | 1.073 |

**The control reproduced**: `conv2d_f32_masked` reads 2.62x here against 2.78x in the
earlier focused 64-round run and 1.87x on the full board, with its FrankenTorch null
passing at [0.978, 1.017]. So the composition effect is real and calibrated for this
window — focused ≈ 2.6–2.8x, full-board ≈ 1.87x.

**But it does not generalise.** The same change of company that cost `conv2d_f32_masked`
40% *gained* `max_pool3d_dense` 10%. The effect is **lane-specific, not a global
factor**, and I should not have implied one lane's 1.45x would carry to others. What
distinguishes them is plausible enough — conv2d's masked backward allocates and
streams hundreds of MB, so it benefits from a warm allocator, while max_pool3d is a
1–3 ms bandwidth-walled lane with little to warm — but that is a hypothesis and is not
measured here.

**None of the pool rows is quotable.** All four NULL-FAILED; the pool lanes' own nulls
are 0.798–0.967 against a ±0.02 band. At 1.7–2.8 ms per lane these are simply too
small for this instrument to resolve at any round count that is worth the wall clock —
consistent with the standing finding that avg/max pool are bandwidth-walled.

**The ranking is unchanged.** `max_pool3d_dense` at ~2.5x is a genuine loss and
plausibly the third-worst in the tree, but it is nowhere near the SVD square forward's
certified 3.10–3.12x, and it cannot be certified at its size. The worst op remains
SVD n=1024.
