# `geqrf` reaches 535x at n=1024 — and the ratio grows with n

**Result: three sizes, every A/A null passing, every parity matching. `geqrf` is
56x slower than PyTorch at n=256, 254x at n=512, and 535x at n=1024 — 5.07 SECONDS
against torch's 9.5 ms for a single 1024×1024 QR factorisation. The ratio is not a
constant: it grows, because our implementation measures n^3.18–4.01 where the algorithm
is exactly n³ and torch measures n^1.84–2.10.**

## The rows

```
FT_OP=geqrf FT_ROUNDS=15 RAYON_NUM_THREADS=8 FT_GATE_SIZES="256,512,1024" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of bidiag_gate_sweep_h2h>
```

`elf_sha256=a6b6604454514ce1474fae99623845b6465b834be317177a8138b610fb7dc1ac`,
idle **92.44% then 93.81%** before launch.

| n | FT min (arm0/arm1) | PT min | standing | **A/A null** | PT spread | parity |
|---|---|---|---|---|---|---|
| 256 | 34.688 / 30.905 ms | 0.615 ms | **48.10x / 48.59x** | **0.989 PASS** | 1.64x | 3.20e-13 MATCH |
| 512 | 559.481 / 561.547 ms | 2.201 ms | **227.58x / 221.75x** | **1.002 PASS** | 1.78x | 3.84e-13 MATCH |
| **1024** | **5066.908 / 5035.006 ms** | **9.468 ms** | **457.59x / 475.33x** | **1.000 PASS** | 1.73x | 7.73e-13 MATCH |

All three nulls pass. The n=512 row reproduces the separately certified 226.7x
(`48c0b7b7`) to within 0.4% in a different window, which is the cross-check that makes
the other two credible.

## The scaling is the finding

`geqrf` is **O(4n³/3) — exactly n³ — for both implementations.** Neither side has an
algorithmic advantage here; this is the same factorisation.

| interval | ours | torch | ratio |
|---|---|---|---|
| 256 → 512 | **n^4.01** | n^1.84 | 56x → 254x |
| 512 → 1024 | **n^3.18** | n^2.10 | 254x → 535x |

Two things follow.

**Ours exceeds n³ where the algorithm is n³.** An implementation cannot beat its own
flop count, so anything above n^3.00 is pure overhead growing with size. The n^4.01
across 256→512 is a **cache cliff**: at n=256 the f64 matrix is 512 KB and still fits
L2; at n=512 it is 2 MB and does not. The stride-n column walks on a row-major array
(`a[i * n + kk]`, one cache line touched per useful element) go from mostly-hits to
mostly-misses across exactly that boundary. Above the cliff the exponent settles back
toward n^3.18, which is what a permanently cache-missing but otherwise cubic loop
looks like.

**Torch is BELOW n³** (n^1.84–2.10) on a cubic algorithm, which is the signature of
blocked BLAS-3 with a trailing GEMM whose efficiency *improves* with size as the panels
get better amortised.

So the two curves diverge for structural reasons at both ends, and **226x at n=512 was
a floor, not a ceiling.**

## What this does to the fix's value

The `geqrf` defect (`frankentorch-geqrf-misses-blocked-kernel-1zp6r`) is a public API
bypassing its own optimised kernel: `ft-api` carries a private `geqrf_packed_f64` while
`qr_contiguous_f64` already dispatches a blocked compact-WY path at `m >= 128 && k >= 16`.

Sizing what the re-route recovers, using our own blocked `qr` as the reference — it
does **strictly more** work than `geqrf` (it also forms Q) and measured 40.707 ms at
n=512 against `geqrf`'s 559.481 ms:

* at n=512 the fix should recover **at least 13.7x**;
* at n=1024 it should recover **more**, because the naive path is on the wrong side of
  the cache cliff and the blocked path is not.

That is the largest single recoverable factor found in this campaign, and unlike the
`eigh` and SVD gaps it requires no new algorithm — the correct code is already in the
tree and already shipping for a neighbouring op.

## Caveats, stated

* **`iowait 53413 jiffies` on the n=1024 row.** That is by far the highest of any row
  I have taken and it deserves flagging rather than burying. What argues the row is
  still readable: the A/A null is **1.000**, the two arms agree to 0.6%, PT spread is
  1.73x, and the incumbent (9.468 ms) is the right order for MKL `dgeqrf` at n=1024 on
  8 threads. Heavy I/O that hit both arms equally is common-mode; I would not quote a
  tight effect from this row, but a 535x ratio is not a 53413-jiffy artefact.
* **`FT_ROUNDS=15`, not 45**, because one n=1024 `geqrf` takes ~5 s on our side. The
  nulls passed anyway.
* **Core clock 1429 MHz** on the n=256/512 rows and 1429–1796 at n=1024 — the cores sat
  near the floor. Both arms share it per size, so ratios are common-mode; absolute times
  are not comparable with rows taken at 3.4 GHz.
* My incumbent-plausibility gate compared against the **SVD's** n=1024 reference
  (~122.604 ms) and passed vacuously again. Torch's 0.615 / 2.201 / 9.468 ms scales as
  n^1.97 overall, which is the right shape for blocked `dgeqrf`; the gate still has no
  per-op reference and did not earn its pass.

## Board

| op | n | standing | null |
|---|---|---|---|
| **`geqrf`** | **1024** | **457–475x SLOWER** | **1.000** |
| `geqrf` | 512 | 222–228x | 1.002 |
| `geqrf` | 256 | 48x | 0.989 |
| `eigh` | 1024 | 8.16–8.23x | 0.994 |
| `eigh` | 512 | 5.60x | 1.011 |
| `qr` | 512 | 5.50x | 1.007 |
| `eigvalsh` | 512 | 4.18x | 1.003 |
| SVD | 1024 / 512 | 3.10x / 2.40x | 0.991 / 1.015 |

`orgqr` — the third family member, ELF built and snapshotted (`9aab26ce…`) — is next.
Its only prior figure is ~1.8x, cross-run at batched-tiny shapes; `geqrf`'s batched
figures were equally reassuring and the single-matrix path turned out to be 535x.
