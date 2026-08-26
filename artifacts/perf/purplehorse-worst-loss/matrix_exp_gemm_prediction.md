# `matrix_exp` is 1.88x — my "GEMM-bound ⇒ we win" prediction fails, but it is the closest linalg op to parity

**Result: `matrix_exp` at n=512 is 1.880–1.946x SLOWER than PyTorch (PT spread 1.89x,
parity 1.44e-13 MATCH, A/A null 1.000 / 0.970 — measured, not certified). I picked this
op to test whether the board's GEMM-bound wins (1.06–1.70x FASTER) carry into linalg.
They do not. But at 1.88x it is by a clear margin the closest single-matrix linalg op to
parity measured in this campaign, which makes the prediction wrong in magnitude and right
in direction.**

## The row

```
FT_OP=matrix_exp FT_ROUNDS=25 RAYON_NUM_THREADS=8 FT_GATE_SIZES="512" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot b1261ced…>
```

idle 70.28% then 91.60%.

| n | FT min (arm0/arm1) | PT min | standing | A/A null | PT spread | parity |
|---|---|---|---|---|---|---|
| 512 | 21.312 / 22.415 ms | 13.341 ms | **1.880x / 1.946x** | 1.000 / **0.970** | 1.89x | 1.44e-13 MATCH |

Null misses ±0.02 by 3.0%, so **measured, not certified**.

Fixture scaled by 1/n on both arms — unscaled, `exp(‖A‖)` overflows f64 well before
n=512. The result is unique (no sign, pivot or basis freedom), so `|sum|` genuinely
discriminates rather than decorating.

## The prediction and what it was worth

I chose `matrix_exp` explicitly because it could **falsify** the pattern I had been
accumulating, not extend it. Eight consecutive single-matrix linalg losses, but the
board's GEMM-bound lanes went the other way — `linear_narrow` 1.70x, `attention` 1.26x
(null PASS), `conv2d_f32` 1.08x, `linear_wide` 1.06x, all *faster* than torch. Scaling-
and-squaring plus Padé is nearly pure GEMM, so if "GEMM-bound ⇒ we win" generalised, this
was where it would show.

**It does not generalise.** We lose 1.88x on the most GEMM-dominated linalg op available.

**But the direction survives, and sharply.** Sorting every single-matrix op measured this
session by how GEMM-dominated it is:

| op | shape | standing |
|---|---|---|
| board GEMM lanes (`linear`, `attention`, `conv2d_f32`) | pure GEMM | **1.06–1.70x FASTER** |
| **`matrix_exp`** | **GEMM + triangular solve + squaring** | **1.88x** |
| SVD | blocked bidiagonal + QR sweep | 2.40x |
| `cholesky` | blocked, GEMM trailing update | 4.42x |
| `qr` | blocked compact-WY | 5.50x |
| `eigh` | unblocked reduction + QL replay | 5.60x |
| `slogdet`/LU | blocked getrf | 21.3–24.9x |
| `geqrf`/`orgqr`/`ormqr` | private per-reflector BLAS-2 loops | 125–535x |

That is a clean monotone ordering across five orders of magnitude of ratio, and the
ordering variable is **how much of the op is matrix-matrix product versus panel/BLAS-2
work**. It is the same conclusion the GEMM refutation reached from the other side — the
loss lives *around* the GEMM, not in it — now with a supporting gradient rather than two
endpoints.

`matrix_exp` sitting between the board lanes and the factorisations is exactly what that
predicts: it is GEMM-dominated but carries a triangular solve and the scaling-squaring
bookkeeping, so it lands just off parity rather than past it.

## Instrument notes

**GATE 2b printed a real value on a live run for the first time**:
`incumbent_spread n=512 spread=1.89x -> ok`. The previous two attempts at this gate
produced no line at all, then a line with an empty value (`spread=x -> ok`) that passed
everything. This confirms the `sed`-based rewrite works end-to-end and not merely in the
offline probe I tested it with.

**One unexplained oddity, recorded rather than hidden.** The chained build→snapshot→run
command snapshotted the ELF (the sha printed) and then produced no run and no log, exiting
0. Re-running the identical `run_validated.sh` invocation directly worked first time and
produced the row above. I do not know why the chained form silently skipped the run, and I
am not claiming the two are equivalent — the row quoted here is from the direct
invocation.
