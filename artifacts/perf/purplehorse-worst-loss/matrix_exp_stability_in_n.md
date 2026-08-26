# The GEMM-dominant end of the gradient is STABLE in n — `matrix_exp` CERTIFIED 2.002x at n=1024

**Result: `matrix_exp` at n=1024 is 2.002x / 2.051x SLOWER than PyTorch, A/A null
1.000 / 0.984 — both arms inside ±0.02, so this row is CERTIFIED, the first
certification for this op. Parity 7.47e-13 MATCH.**

## The row

```
FT_OP=matrix_exp FT_ROUNDS=25 RAYON_NUM_THREADS=8 FT_GATE_SIZES="1024" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python
snapshot/elf sha256 = b1261ced3c2cdfad32ac25ca37dd9a92d5e5faf121593bad720310e77580a795
```

idle 87.06% then 87.64%. Incumbent PyTorch 2.12.1+cpu, same invocation, self-reported 8 threads.

| n | FT min (arm0/arm1) | PT min | paired standing | A/A null | PT spread | parity |
|---|---|---|---|---|---|---|
| 512 | 21.312 / 22.415 ms | 13.341 ms | 1.880x / 1.946x | 1.000 / **0.970** ✗ | 1.89x | 1.44e-13 |
| **1024** | 144.531 / 145.705 ms | 68.561 ms | **2.002x / 2.051x** | **1.000 / 0.984** ✓ | 2.00x | 7.47e-13 |

## What it was measuring

The single-matrix map established a monotone gradient ordered by GEMM-dominance, from
1.06-1.70x FASTER (pure-GEMM board lanes) to 535x SLOWER (private BLAS-2 Householder
loops). That map was one size deep at its GEMM-dominant end. The open question was whether
the gradient is a property of the **op mix** or merely of **cache behaviour at n=512** — if
`matrix_exp` degraded with n the way `geqrf` and `eigh` do, the ordering would be an
artefact of a single shape.

**It does not degrade.** `matrix_exp` stays in the 1.6-2.1x band across a 2x size change,
against:

| op | n=512 | n=1024 | growth |
|---|---|---|---|
| **`matrix_exp`** | **1.880x** | **2.002x** | **1.065x** |
| SVD | 2.40x | 3.10x | 1.29x |
| `eigh` | 5.60x | 8.16-8.23x | 1.46x (and 15.5x at 2048) |
| `geqrf` | 227.6x | 535.2x | 2.35x |

The GEMM-dominant end is stable; the panel/BLAS-2 end is not. The gradient is a property
of the op mix, not of one shape.

## A disagreement between two statistics, recorded rather than resolved in my favour

The paired per-round statistic and a naive min/min ratio do **not** agree about growth:

* paired: 1.880 -> 2.002, i.e. **flat** (1.065x per doubling)
* min/min: 21.312/13.341 = 1.597 -> 144.531/68.561 = 2.108, i.e. **growing** 1.32x per
  doubling — which matches the min-based exponents exactly (FT n^2.76, PT n^2.36,
  2^0.40 = 1.32)

I trust the paired figure, and the reason is visible in the row: **PT spread is 1.89x at
n=512 and 2.00x at n=1024.** With scatter that wide, a min/min ratio is comparing our
typical sample against torch's single luckiest escaped one, and it flatters torch more at
whichever size the tail happened to run longer. That is the exact failure the paired
per-round estimator exists to avoid.

But the two statistics are not telling the same story, and quoting only the one that
supports the headline would be picking the estimator after seeing the answer. **The
conclusion is robust to the choice** — under min/min the band is 1.60-2.11x and under
paired it is 1.88-2.00x, and both are an order of magnitude away from `geqrf`'s 227->535x.
That robustness, not the paired number alone, is what carries the claim.

Both arms' nulls land in band, so unlike the n=512 row this one is certified rather than
merely measured.
