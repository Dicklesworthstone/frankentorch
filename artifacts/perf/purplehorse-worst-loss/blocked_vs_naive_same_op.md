# The same factorisation, two code paths in one tree: 8x vs 535x

**Result: `qr` (blocked kernel) and `geqrf` (private naive loop) compute the same
Householder QR. Measured against the same live incumbent on the same host, `qr` is
3.63x → 8.10x slower than PyTorch across n=256→1024 while `geqrf` is 56x → 535x. Our
blocked path scales n^2.57–2.59; our naive path scales n^3.18–4.01 on an algorithm that
is exactly n³. The defect is not that we lack fast code — it is that a public API does
not call the fast code sitting next to it.**

## One line that states the whole bead

At n=512, all three measured against a live torch co-process:

| path | time | vs torch |
|---|---|---|
| `tensor_linalg_qr` — **blocked kernel** | **40.707 ms** | 5.50x |
| `tensor_geqrf` — naive private loop | 559.481 ms | 227.58x |
| `tensor_orgqr` — naive private loop | 777.959 ms | 125.19x |
| `geqrf` + `orgqr` | **1337.440 ms** | — |

**Our own `qr` is 32.9x faster than our own `geqrf` + `orgqr`, computing the same
factorisation.**

## The scaling, side by side

```
FT_OP=qr FT_ROUNDS=25 RAYON_NUM_THREADS=8 FT_GATE_SIZES="256,1024" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of bidiag_gate_sweep_h2h>
```

`elf_sha256=75c33aa3d94b5ee1cbf20ab4d3ae689b...`, idle **75.81% then 91.63%**.

| n | qr FT | qr PT | **qr ratio** | qr null | geqrf FT | geqrf PT | **geqrf ratio** |
|---|---|---|---|---|---|---|---|
| 256 | 6.879 ms | 1.897 ms | **3.63x** | **0.999 PASS** | 34.688 ms | 0.615 ms | **56.4x** |
| 512 | 40.707 ms | 6.441 ms | **6.32x** | 1.007 PASS | 559.481 ms | 2.201 ms | **254.2x** |
| 1024 | 244.883 ms | 30.238 ms | **8.10x** | 1.000 / **1.067 FAIL** | 5066.908 ms | 9.468 ms | **535.2x** |

Every row's PT spread passed the newly-landed GATE 2b (1.48x, 1.35x here; the gate
reported `ok` on both). Parity MATCH throughout.

**n=1024 `qr` is measured, not certified** — its second arm's null is 1.067, missing the
±0.02 band by 6.7%. The n=256 row passes at 0.999 and n=512 was separately certified at
1.007, so the trend is sound even though that one cell is not bankable.

### Exponents — the algorithm is n³ for both paths

| interval | qr (blocked) | geqrf (naive) | torch |
|---|---|---|---|
| 256 → 512 | **n^2.57** | **n^4.01** | n^1.76 / n^1.84 |
| 512 → 1024 | **n^2.59** | **n^3.18** | n^2.23 / n^2.10 |

**Ratio growth over a 4x size increase: `qr` 2.2x, `geqrf` 9.5x.**

Our blocked path comes in *below* n³ — the same signature torch shows, and what blocked
BLAS-3 does as its trailing GEMM amortises better with size. Our naive path comes in
*above* n³, which an implementation cannot do on its own flop count; the excess is
overhead growing with size, and the n^4.01 step across 256→512 is the cache cliff where
the f64 matrix goes 512 KB → 2 MB and the stride-n column walks stop fitting L2.

## What this settles

Two causal stories were refuted earlier in this campaign — "we lack blocking" (our QR
*is* blocked) and "our GEMM is the floor" (GEMM lanes beat torch by 1.06–1.70x). This
row pair shows why both were wrong and what is actually true: **the tree contains both a
good implementation and a bad one for the same operation, and which one you get depends
on which public entry point you call.**

* `tensor_linalg_qr` → `qr_contiguous_f64` → blocked compact-WY at `m >= 128 && k >= 16`.
* `tensor_geqrf` → a private `geqrf_packed_f64` in `ft-api` that never leaves the crate.

Nothing about algorithm choice, blocking availability, GEMM quality or hardware explains
the 535x. The fast code is present, shipping, and reachable.

## Sizing the fix, from measured numbers only

`qr` does **strictly more** work than `geqrf` — it also forms Q — and is 13.7x faster at
n=512 and **20.7x** faster at n=1024. So re-routing `geqrf` through the blocked forward
pass should recover at least that, and the margin **widens with n** because the naive
path is on the wrong side of the cache cliff and the blocked path is not.

Bead `frankentorch-geqrf-misses-blocked-kernel-1zp6r` already records that the blocked
kernel computes and then discards exactly what `geqrf` needs (`qr_factor_panel_leaf_f64`
takes `vmat` and `tau` out-parameters; the entry point returns only Q). The open
questions there are unchanged and still not assumed: the packed-V layout our own
`orgqr`/`ormqr` consumers expect, and bit-exactness under the ratified tolerance policy
since the blocked path is documented as not bit-identical above its threshold.

## Board

| op | n | standing | null |
|---|---|---|---|
| `geqrf` | 1024 | **457–475x** | 1.000 |
| `geqrf` | 512 | 222–228x | 1.002 |
| `orgqr` | 512 | 125.19–125.45x | 1.000 / 0.987 |
| `geqrf` | 256 | 48x | 0.989 |
| `eigh` | 1024 | 8.16–8.23x | 0.994 |
| `qr` | 1024 | 8.11–8.30x | 1.000 / 1.067 FAIL |
| `eigh` / `qr` / `eigvalsh` | 512 | 5.60x / 5.50x / 4.18x | all PASS |
| `qr` | 256 | 3.63–3.69x | 0.999 PASS |
| SVD | 1024 / 512 | 3.10x / 2.40x | PASS |

`ormqr` is still building — the last family member without a live number.
