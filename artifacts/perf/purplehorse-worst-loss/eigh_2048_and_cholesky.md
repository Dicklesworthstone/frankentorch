# `eigh` n=2048 confirms my extrapolation — and `cholesky`, picked expecting a win, is a 4.5x loss

Two measurements, chosen to test claims of mine rather than to add ops.

## 1. `eigh` at n=2048 — the extrapolation held

I committed a prediction in `1a31f3c3`: from ours n^2.96 against torch n^2.14, n=2048
should read **~14.5x**, flagged explicitly as *"NOT measured — three points is thin, and
I have already been burned this session by a tidy model that one check refuted."*

**Measured: 15.513x / 16.475x.**

| n | ours | torch | ratio |
|---|---|---|---|
| 256 | 8.298 ms | 2.795 ms | 2.97x |
| 512 | 65.067 ms | 11.903 ms | 5.47x |
| 1024 | 502.695 ms | 54.490 ms | 9.23x |
| **2048** | **5132.756 ms** | **330.770 ms** | **15.52x** |

`elf` = `bidiag_elf_ormqr`, idle 94.49% then 93.76%, PT spread 1.25x, **parity rel
0.00e0 — an exact bit match** on the eigenvalue sum. A/A null 1.000 / **0.970**, missing
±0.02 by 3.0%, so the row is **measured, not certified**.

Exponents per interval:

| interval | ours | torch |
|---|---|---|
| 256 → 512 | n^2.97 | n^2.09 |
| 512 → 1024 | n^2.95 | n^2.19 |
| **1024 → 2048** | **n^3.35** | **n^2.60** |

**Both arms steepened at the last step** — ours 2.96 → 3.35, torch 2.14 → 2.60 — which is
why the prediction landed slightly low rather than badly wrong. At n=2048 an f64 matrix
is 33.5 MB and the QL replay streams the whole `ops` log plus every row, so a
memory-hierarchy effect at that size is the obvious candidate; that it hits torch too
suggests it is the machine rather than our implementation, and I am not claiming
otherwise without measuring it.

The substantive point survives: **the gap is still widening with n and shows no sign of
a ceiling** — 2.97x → 15.52x across a 8x size range. Divide-and-conquer remains the only
lever that touches the exponent.

## 2. `cholesky` — I picked it expecting a win, and we lose 4.5x

This was chosen deliberately against my own narrative: a blocked kernel exists in-tree,
memory records a shipped 2.3x blocked-Cholesky win, the factorisation is *unique* for an
SPD matrix so parity genuinely discriminates, and the ledger calls the direct
factorisations MKL-batched walls. If anything in this session was going to come back near
parity, it was this.

```
FT_OP=cholesky FT_ROUNDS=25 RAYON_NUM_THREADS=8 FT_GATE_SIZES="512,1024" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot 2187280c…>
```

Fixture is symmetrise-then-add-`n`-to-the-diagonal on **both** arms — strictly
diagonally dominant, hence positive definite at every n. Without it the existing fixture
is not SPD at n=512 (off-diagonal row mass ~100 against a diagonal of 3) and torch would
simply reject it.

| n | FT min (arm0/arm1) | PT min | standing | A/A null | PT spread | parity |
|---|---|---|---|---|---|---|
| **512** | 5.167 / 5.157 ms | 1.177 ms | **4.484x / 4.417x** | **1.000 / 0.982 PASS** | 2.40x | 2.55e-12 MATCH |
| 1024 | 21.857 / 20.538 ms | 3.713 ms | 5.701x / 5.750x | 1.000 / **1.034** | 1.26x | 5.78e-13 MATCH |

**n=512 is CERTIFIED at 4.42–4.48x SLOWER.** n=1024 misses its null by 3.4% and is
measured, not certified.

**My expectation was wrong.** "We do fine on the direct factorisations" does not survive
contact with a live incumbent on the single-matrix path. The blocked kernel is real and
is being called — my entry-point sweep confirmed `tensor_linalg_cholesky` reaches
`ft_kernel_cpu` — and we are still 4.5x behind.

### But it does constrain the story, which is what it was for

| path class | ops | standing |
|---|---|---|
| **naive private loops** | `geqrf`, `orgqr`, `ormqr` | **125–535x** |
| **blocked kernels** | `qr`, `cholesky` | **4.4–8.3x** |
| unblocked reduction + QL | `eigh` | 3.0–16.5x, growing |
| blocked bidiagonal | SVD | 2.4–3.1x |

Two orders of magnitude separate "calls its blocked kernel" from "carries a private
loop". That is the sharpest form of the `geqrf` bead: the defect is not that our blocked
code is bad — it is 4.4–8.3x, which is a normal gap against MKL — it is that three public
entry points never reach it.

Cholesky's scaling supports the same reading: ours n^2.08, torch n^1.66, both well below
the algorithm's n³, and the ratio grows only 1.27x per doubling against `geqrf`'s 2.35x.
That is what a blocked implementation looks like even when it is losing.

## Method notes

* The plausibility gate **skipped** both ops — no `eigh` n=2048 reference and no
  `cholesky` reference at all — which is the honest default I built rather than
  borrowing another op's figure. Torch's numbers were sanity-checked against their own
  scaling instead: `eigh` 54.490 → 330.770 ms is n^2.60, `cholesky` 1.177 → 3.713 ms is
  n^1.66, both plausible for LAPACK.
* Two build failures on the way, both from guessing a signature off the torch API rather
  than reading ours: `tensor_diagonal(r, 0, 0, 1)` (takes `(input, offset)`) and
  `tensor_linalg_cholesky(x)` (takes `(input, upper)`). The compiler caught both, but
  each cost a ~14-minute remote build. Reading the signature first is strictly cheaper.
