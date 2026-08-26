# `slogdet`/LU is 21.3x — outside the band I drew, and it exposed a gate that never fired

**Result: `slogdet` at n=1024 is 21.314–21.626x SLOWER than PyTorch (A/A null 1.000 /
1.010 PASS, PT spread 1.33x, parity 3.75e-13 MATCH) — CERTIFIED. That is 3–5x worse than
the 4.4–8.3x "blocked kernel" band I proposed one cycle ago, and our LU *is* blocked. The
band is refuted. Separately, the n=512 row of the same run was reported valid despite a
PT spread of 3.67x, because GATE 2b was nested inside the per-op reference loop and an op
with no reference got no spread check at all.**

## The rows

```
FT_OP=slogdet FT_ROUNDS=25 RAYON_NUM_THREADS=8 FT_GATE_SIZES="512,1024" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot 13355d7d…>
```

`elf` built on `vmi1153651`, `ldd` clean. idle 74.18% then 83.76%.

| n | FT min (arm0/arm1) | PT min | standing | A/A null | PT spread | parity |
|---|---|---|---|---|---|---|
| 512 | 35.962 / 29.083 ms | 1.333 ms | 26.44x / 25.97x | 1.000 / 0.986 | **3.67x — VOID** | 3.28e-13 MATCH |
| **1024** | **106.005 / 93.418 ms** | **4.950 ms** | **21.31x / 21.63x** | **1.000 / 1.010 PASS** | **1.33x** | 3.75e-13 MATCH |

**n=1024 is CERTIFIED. n=512 is void** on spread — see the gate bug below, which is why it
was not voided automatically.

`slogdet` was chosen to price the **LU path** with an unambiguous checksum: LU's row
permutation can legitimately differ between implementations, so a factor-level checksum
would mismatch on pivot order rather than on a defect; `logabsdet` is invariant to both
pivot order and sign. Plain `det` was unusable because the SPD fixture's determinant is
~n^n, not representable in f64 at n=512.

## The band I proposed is refuted

One cycle ago, after `cholesky` came in at 4.42–4.48x, I wrote that two classes separate
by two orders of magnitude:

| class | ops | claimed band |
|---|---|---|
| naive private loops | `geqrf`, `orgqr`, `ormqr` | 125–535x |
| **blocked kernels** | `qr`, `cholesky` | **4.4–8.3x** |

**LU is blocked** — `ft-kernel-cpu` line 24030 documents "the same blocked right-looking
`getrf` scheme" — and it measures **21.3x**. That is 3–5x outside the band.

So the tight "blocked kernels sit at 4.4–8.3x" claim does not survive its third test. The
corrected picture:

| class | ops | measured |
|---|---|---|
| reaches a kernel | `cholesky` 4.5x, `qr` 5.5–8.3x, **`slogdet`/LU 21.3x** | **4.4–21.6x** |
| private loop in `ft-api` | `geqrf`, `orgqr`, `ormqr` | **125–535x** |

**What survives is the separation, not the tightness.** An order of magnitude still
divides "reaches its kernel" from "carries a private loop", and that is the load-bearing
part of the `geqrf` bead. What does not survive is my implication that blocked code
reliably lands near 5x — being blocked is necessary, not sufficient, and LU is the
counterexample.

That is the third mechanism I have proposed in this campaign and had to withdraw or
narrow. The other two were "we lack blocking" (our QR *is* blocked) and "our GEMM is the
floor" (GEMM lanes beat torch by 1.06–1.70x).

## The gate bug — GATE 2b never ran for this op

`slogdet` n=512 has a PT spread of **3.67x**, over the 3.0 ceiling, and the run reported
`VALIDATED_RUN_OK`. No `incumbent_spread` line appears in its output at all.

Cause: GATE 2b was written *inside* the per-op reference loop:

```sh
while read -r n ref; do
    ...spread check...
    ...plausibility check...
done <<< "$(refs_for_op)"
```

`refs_for_op` deliberately returns **empty** for ops with no banked reference — that was
the honest default, so an absent reference skips the *plausibility* check rather than
borrowing another op's figure. But an empty heredoc means **zero loop iterations**, so the
spread check never executed either. Intended: absent reference → skip plausibility.
Actual: absent reference → skip plausibility **and** spread.

Fixed: GATE 2b now runs as a standalone pass over every size present in the row output,
independent of the reference table.

**This is the same shape as every instrument defect this session** — the success signal
derived from something other than the thing that had to succeed. Previous instances: a
retry loop printing `exit 0` over a `HARD_FAIL`; `VALIDATED_RUN_OK` over a run that
produced zero rows because the gate inspected rows and there were none; an ELF that
compiled cleanly and could not load. Each was found only because a number looked wrong
afterwards, which is not a reliable detector.

## Board

| class | op | n=512 | n=1024 |
|---|---|---|---|
| private loop | `geqrf` | 227.6x | **535.2x** |
| private loop | `ormqr` | ~253x | **502–507x** |
| private loop | `orgqr` | 125.2x | 312–317x |
| **kernel** | **`slogdet`/LU** | *void* | **21.3–21.6x** |
| kernel | `qr` | 5.50x | 8.11–8.30x |
| kernel | `cholesky` | 4.42–4.48x | 5.70–5.75x |
| mixed | `eigh` | 5.60x | 8.16–8.23x (15.5x at n=2048) |
| kernel | SVD | 2.40x | 3.10x |

## The gate fix took two attempts — the first one looked like it worked

Fixing GATE 2b to run outside the reference loop was not enough. The re-run printed:

```
incumbent_spread n=512 spread=x -> ok
```

**The gate fired and validated nothing.** My extraction used `grep -oP` with an
alternation piped through `paste - -`; it emitted the size but an *empty* spread, so
`awk` compared `""` against 3.0 — false — and every row reported `ok`.

**That is strictly worse than the bug it replaced.** The original failure produced *no*
line, which is at least silent. This one produced a reassuring one. Had I not read the
value, I would have recorded "GATE 2b now fires correctly" on the strength of seeing the
line appear.

Rewritten as a single `sed` capturing both fields from one line, so a parse failure
yields no row rather than a half-parsed one, plus an explicit guard that treats a missing
value as a **failure** instead of a pass. Verified two ways rather than by inspection:

* extraction against the real log → `512 3.67` / `1024 1.33`;
* end-to-end probe on the exact row that slipped through →
  `n=512 spread=3.67x -> WILD`, `n=1024 spread=1.33x -> ok`, `spread_bad=1`.

That is the fourth instrument defect this session with the same signature, and the
second where my *fix* had the signature too. The lesson that keeps recurring: **seeing a
check run is not evidence that it checked anything** — the value it computed has to be
looked at.

## `slogdet` n=512, re-taken in a clean window

| n | FT min (arm0/arm1) | PT min | standing | A/A null | PT spread | parity |
|---|---|---|---|---|---|---|
| 512 | 22.999 / 23.023 ms | 1.073 ms | **24.93x / 24.05x** | **1.000 / 0.980 PASS** | 1.77x | 3.28e-13 MATCH |

Spread 1.77x, inside the ceiling — verified by reading the row, not by trusting the gate
that was broken at the time. Both arms agree to 0.1%.

**`slogdet`/LU therefore stands at 24.0–24.9x (n=512) and 21.3–21.6x (n=1024)**, both
with passing nulls — comfortably outside the 4.4–8.3x band and confirming that the band
was the wrong shape rather than n=512 being an outlier.
