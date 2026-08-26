# `inv` CERTIFIED 5.4-7.5x — and it REFUTES my own "slogdet/LU 21-25x" map row

**Result: `linalg.inv` measures 7.549x / 7.120x at n=512 and 5.424x / 5.757x at n=1024,
all four A/A nulls in band (1.000/1.007, 1.000/0.981) — CERTIFIED at both sizes. Parity
8.48e-13 and 1.29e-13 MATCH. ELF
`b822dab411ec4d0cb04d45bf184f8901800bf68ea7e22a7b03e19db1a7b22b8c`.**

## The row

```
FT_OP=inv FT_ROUNDS=25 RAYON_NUM_THREADS=8 FT_GATE_SIZES="512,1024" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python
```
idle 93.55% then 93.52%. Incumbent PyTorch 2.12.1+cpu, same invocation, 8 threads.

| n | FT min (arm0/arm1) | PT min | paired standing | A/A null | PT spread | parity |
|---|---|---|---|---|---|---|
| 512 | 21.021 / 20.749 ms | 3.382 ms | **7.549x / 7.120x** | 1.000 / 1.007 | 1.49x | 8.48e-13 |
| 1024 | 82.520 / 79.085 ms | 15.937 ms | **5.424x / 5.757x** | 1.000 / 0.981 | 1.33x | 1.29e-13 |

## What it was measuring, and the answer

The map carried one row reading "`slogdet`/LU — blocked getrf — 21.3-24.9x". That row
names an op and a *route* together, and I had never checked that the route was what cost
the 21x. `inv` is LU-backed too but replaces slogdet's O(n) diagonal log-product with an
O(n^3) getri tail, so it separates the two readings.

**My grouping was wrong.** `inv` does strictly more work than `slogdet` and lands 3.2-3.9x
BETTER against torch. So "the LU family sits at 21-25x" is false; only `slogdet` does.

## Both ends of the ratio moved — and the informative end is ours

A ratio has a numerator and a denominator, and the interesting movement here is in FT's
own absolute times, not torch's. At n=1024:

| | FT min | PT min | ratio |
|---|---|---|---|
| `slogdet` (getrf + O(n) tail) | 93.418 / 106.005 ms | 4.950 ms | 21.3-21.6x |
| `inv` (getrf + O(n^3) getri) | 79.085 / 82.520 ms | 15.937 ms | 5.4-5.8x |

**Torch is the sensible way round**: its slogdet is 3.2x faster than its inv, exactly as
the flop counts predict. **We are backwards**: our slogdet is 1.13-1.18x SLOWER than our
own inv while doing strictly less work.

That is the same shape as the Householder proof ("our own `qr` is 32.9x faster than our own
`geqrf`+`orgqr`") — an internal inconsistency that needs no reference implementation to
read as a defect.

## Mechanism, found in source and stated as a falsifiable prediction

`tensor_linalg_slogdet` (ft-api/src/lib.rs) calls `slogdet_contiguous_f64` **twice per
forward on the same matrix**: once to take `.sign` (discarding `logabsdet`), then again
inside the autograd forward closure to take `.logabsdet` (discarding `sign`). One result
struct carries both fields. We factorise twice and throw half of each away.

The second call is legitimate when a tape node is needed — the closure must be replayable
under `create_graph`. With no grad there is no closure to replay, so both outputs can come
off one call. Bit-exact by construction: same kernel, same input.

**Prediction committed before measuring:** a no-grad fast path drops slogdet n=1024 from
93-106 ms to ~50-55 ms, i.e. 21.3x -> ~10-11x. If it barely moves, this reading is wrong
and I will say so. Source reading has produced three confident wrong answers this campaign,
so the fix's own measurement is the proof, not the reading.

## Map correction

The single-matrix map's LU row must be split:

| was | is |
|---|---|
| `slogdet`/LU — 21.3-24.9x | `inv` (LU + getri) — **5.4-7.5x CERTIFIED** |
| | `slogdet` — 21.3-24.9x, and anomalously slower in absolute ms than our own `inv` |

`inv` at 5.4-7.5x sits squarely in the established "reaches a kernel" band (4.4-8.3x)
alongside cholesky/qr/eigh. `slogdet` sits above it for a reason specific to slogdet.
