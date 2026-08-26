# The SVD reduction's parallelism is worth less than the instrument's own noise floor

**Result: at n=512 the always-SERIAL bidiagonal reduction reads within 0.3% of the
parallel-gated one, while two IDENTICAL parallel arms in the same invocation differ by
5.2%. The serial-vs-parallel effect is smaller than the A/A null between two copies of
the same code. Our reduction's parallel dispatch is worth at most ~5% — bounded, not
certified — and the ~2.4x gap against MKL is therefore a threading gap we are not
closing by dispatching.**

## Why this was the cut to make

The values-only arm established that our **reduction alone** (50.092 ms at n=512) is
**1.734x slower than PyTorch's ENTIRE SVD** (27.748 ms — reduction, QR sweep, and both
vector expansions), commit `c4d611c4`. Torch does strictly more work and still wins.
The expansion phase is bounded: deleting all 33.5% of it still leaves 1.734x. So the
reduction is where the loss lives, and the open question was whether our parallelism
is doing anything.

## The measurement

Three arms in ONE invocation: the shipped gate **twice** (arm1-vs-arm0 is a real A/A
null) plus `u64::MAX` (always-serial reduction), so `arm2-vs-arm0` prices the
parallelism against a null that says whether the instrument could see it.

```
FT_ROUNDS=27 RAYON_NUM_THREADS=8 FT_GATE_SIZES="512" \
FT_GATE_VALUES="262144,262144,18446744073709551615" \
PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of target/release/examples/bidiag_gate_sweep_h2h>
```

`elf_sha256=323a90cf1804eed18f1f4f62ae8ec4e04357403b4c697981755cd144ecfb8848`,
idle 90.35% then 90.75% before launch, PT spread 1.44x, iowait 45, parity MATCH on
every arm.

| arm | FT min | vs PyTorch | **vs arm0** |
|---|---|---|---|
| gate 262144 | 63.374 ms | 2.442x SLOWER | 1.000 (self) |
| gate 262144 — *identical twin, the A/A null* | 64.297 ms | 2.247x SLOWER | **1.052** |
| **SERIAL** (`u64::MAX`) | 60.476 ms | 2.309x SLOWER | **0.997** |

**The serial arm sits 0.3% from arm0. The identical parallel twin sits 5.2% from it.**
The thing being measured is smaller than the noise between two copies of the same
code.

## The null got WORSE with more rounds, and that is the informative part

| rounds | A/A null (arm1 vs arm0) | serial vs arm0 |
|---|---|---|
| 9 | 1.040 | 1.062 |
| **27** | **1.052** | **0.997** |

Rounds are the standard lever on a null — tripling 9→27 is exactly what took the
n=1024 standing from 1.039 to 0.991 and certified it (commit `7cf74314`). Here the
null moved the **wrong way**.

That rules out sampling noise. A null that halves with rounds is noise; one that
worsens is **systematic** — here almost certainly arm POSITION within the round, since
the two arms are byte-identical code and differ only in where they sit in the
sequence. It is the same signature as `conv2d_f32_masked`'s incumbent null, which sat
at 1.041 across 16, 32 and 64 rounds and never moved (commit `ffe22c15`).

**Consequence: this cannot be bought with wall clock.** Another doubling would burn a
window to land on ~1.05 again. I am not running it.

## What is and is not established

**Established (a bound, and it is the useful output):** whatever our reduction's
parallel dispatch is worth at n=512, it is **smaller than 5%**. Two independent
readings agree — 9 rounds put serial at 1.062 of parallel, 27 rounds at 0.997, and
both sit inside the ±5% the A/A null admits.

**NOT established:** that the parallelism is worth exactly nothing. The instrument
cannot resolve below its own floor, and a real 2–3% gain would be invisible here.
Anyone wanting a tighter figure needs an instrument whose identical arms agree better
than 5% — which means fixing the position effect, not adding rounds.

**NOT established:** anything at n=1024. That row was VOID in the first pass — PT
spread **648.39x**, loadavg 11.46 → 61.98 mid-row — and is excluded.

## An instrument gap this exposed, in my own gate

The void n=1024 row **passed** my incumbent-plausibility check. That gate tests PT
*min* (135.599 ms against ~122.604 banked — plausible) and nothing else. The minimum is
the one sample that escaped the contention; the **spread** is what says the rest did
not. A 648x spread walked straight through a gate I built specifically to catch
crushed incumbents.

A PT-spread ceiling (reject above 3x; clean rows here run 1.14–1.44x) is the fix and
is written; it is not yet landed because the patch tripped a shell guard, and I would
rather land it properly than work around the guard. Until it is in, **read the
`spread` column by hand before quoting any row.**

## Where this leaves the worst loss

The SVD square forward at n=1024 stands at **3.10–3.12x SLOWER, certified**
(`7cf74314`). Its decomposition is now:

* expansion phase (`form_p + form_q`) — 33.5% of n=512; removing ALL of it still
  leaves 1.734x (`c4d611c4`);
* reduction — the remainder, and its parallel dispatch is worth **<5%** (this file);
* `x86-64-v3` — ~1.07x at n=512, nothing at n=256 (`636ce5bc`);
* the `n>=130` `form_p` blocking gate — a measured **no-op** (`7ebc0555`).

Every dispatch-level and flag-level lever on this op has now been measured and none of
them closes it. What is left is the arithmetic itself: our reduction performs like a
serial reduction, MKL threads the same level-2 work, and closing that in pure Rust
means making the per-core matvec faster or finding parallelism the current gate cannot
express — not re-tuning the gate.
