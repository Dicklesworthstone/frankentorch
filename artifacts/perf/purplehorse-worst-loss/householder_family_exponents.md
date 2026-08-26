# All three Householder primitives scale n^3.17–3.22 — identically, refuting my locality claim

**Result: the family is complete at two sizes with clean rows. `ormqr` at n=1024 is
502.5–507.5x SLOWER (A/A null 0.996 PASS, PT spread 1.12x, parity 1.54e-12 MATCH) —
the certified row that three earlier attempts could not produce. `orgqr` at n=1024 is
312.2–317.4x. And all three naive paths scale at n^3.17, n^3.18, n^3.22 — statistically
indistinguishable — while torch's scale n^1.90–2.10. That kills the operand-locality
explanation I proposed for the ratio spread.**

## The rows

`elf` = `bidiag_elf_ormqr` (built `vmi1156319`, `ldd` clean).

| op | n | FT min (arm0/arm1) | PT min | standing | A/A null | PT spread | parity |
|---|---|---|---|---|---|---|---|
| **`ormqr`** | 1024 | 7198.162 / 7133.800 ms | 13.744 ms | **502.5–507.5x** | **1.000 / 0.996 PASS** | **1.12x** | 1.54e-12 |
| `orgqr` | 1024 | 6994.576 / 7124.193 ms | 22.054 ms | 312.2–317.4x | 1.000 / **0.974** | 1.12x | 3.02e-13 |

`ormqr` n=1024 is **certified**. `orgqr` n=1024 misses its null by 2.6% and is measured,
not certified.

**The `ormqr` spread problem was real and it resolved by waiting, not by relaxing.**
Three earlier attempts read spreads of 3.54x, 20.86x and 3.56x and were all discarded by
GATE 2b. Attempt 3 at n=1024 landed at **1.12x** — the cleanest incumbent in the family.
I considered raising the ceiling for this op and did not; the honest fix was a longer op
and a better window, and it worked. Had I relaxed the gate I would have banked a number
from a 20.86x-spread row.

## The exponents — the finding

| path | n=512 → 1024 | vs algorithm (n^3.00) |
|---|---|---|
| `geqrf` (naive) | **n^3.18** | above |
| `orgqr` (naive) | **n^3.17** | above |
| `ormqr` (naive) | **n^3.22** | above |
| torch `geqrf` | n^2.10 | below |
| torch `orgqr` | n^1.90 | below |
| our `qr` (blocked) | n^2.59 | below |

**All three of our naive primitives sit at n^3.17–3.22 — a 0.05 spread.** Every torch
path and our own blocked path sit below n³. An implementation cannot beat its own flop
count, so anything above n^3.00 is overhead that grows with size; everything below it is
a blocked implementation whose trailing GEMM amortises better as panels grow.

## What this refutes — my own mechanism, for the second time

I proposed that `orgqr`'s lower ratio (125x vs `geqrf`'s 227x at n=512) came from operand
locality: `orgqr`'s inner loop sweeps one operand contiguously while `geqrf` has both
strided, so `orgqr` should degrade *less* steeply with size.

**It does not.** `orgqr` measures n^3.17 against `geqrf`'s n^3.18 — indistinguishable.
Locality does not touch the exponent.

The corrected statement, which the numbers do support: **the stride-n access pattern sets
the exponent (n^3.17–3.22, uniformly), and operand locality shows up only in the
constant.** Our three ops differ by at most 1.44x in absolute time at n=1024 (5067 /
6995 / 7198 ms) while their ratios differ by 1.7x — and I showed earlier that the ratio
spread is mostly torch's doing, since torch's own three ops differ by 2.3x while ours
differ by 1.44x.

This is the second mechanism I have proposed in this family and had to withdraw. The
first — that `ormqr` would land *between* `orgqr` and `geqrf` — was refuted by it landing
above both. Both were plausible, both fit the data I had at the time, and both were
wrong. What survives is the part that was measured rather than reasoned: the exponent.

## The family, complete

| op | n=512 | n=1024 | our exponent |
|---|---|---|---|
| `geqrf` | 227.6x (null 1.002) | **535.2x** (null 1.000) | n^3.18 |
| `orgqr` | 125.2x (null 1.000/0.987) | 312–317x (null 0.974) | n^3.17 |
| `ormqr` | ~253x (gate-void) | **502–507x** (null 0.996) | n^3.22 |
| **`qr` (blocked)** | **5.50x** (null 1.007) | 8.11–8.30x (null 1.067) | n^2.59 |

Our three naive primitives cost **19.3 seconds** combined at n=1024. Torch's cost
**45.3 ms**. Our own blocked `qr` — which does strictly more work than `geqrf`, since it
also forms Q — costs 244.9 ms.

## Instrument debt closed this cycle

The incumbent-plausibility gate compared **every** op against the SVD's banked figures,
so it passed vacuously for `eigh`, `qr`, `geqrf`, `orgqr` and `ormqr` — I flagged that on
every row as "did not earn its pass". It now carries per-op references, taken from this
session's own clean windows:

```
svd   128 1.369 / 136 1.443 / 256 6.942 / 512 30.037 / 1024 118.918
eigh  256 2.795 / 512 10.240 / 1024 54.490      eigvalsh 512 7.282
qr    256 1.897 / 512 6.441  / 1024 30.238      orgqr    512 5.899
geqrf 256 0.615 / 512 2.201  / 1024 9.468       ormqr    512 2.687 / 1024 14.299
```

`cholesky` deliberately gets **no** reference — an absent one skips the check, which is
honest, rather than borrowing another op's and producing a pass that means nothing.
