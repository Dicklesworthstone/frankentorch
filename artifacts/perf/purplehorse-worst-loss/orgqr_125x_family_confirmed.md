# `orgqr` is 125x SLOWER — the Householder family defect confirmed on a second op

**Result: `orgqr` at n=512 is 125.191–125.450x SLOWER than PyTorch, A/A nulls 1.000 and
0.987, PT spread 1.77x, parity 8.96e-13 MATCH. That is the third LAPACK Householder
primitive measured and the second with a live number, and it refutes the only prior
figure for this op (~1.8x, cross-run at batched-tiny shapes) by nearly two orders of
magnitude on the single-matrix path.**

## The row

```
FT_OP=orgqr FT_ROUNDS=25 RAYON_NUM_THREADS=8 FT_GATE_SIZES="256,512" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of bidiag_gate_sweep_h2h>
```

`elf_sha256=75c33aa3d94b5ee1cbf20ab4d3ae689b...`, idle **83.10% then 77.88%**.

| n | FT min (arm0/arm1) | PT min | standing | A/A null | PT spread | parity |
|---|---|---|---|---|---|---|
| 256 | 38.340 / 50.580 ms | 0.994 ms | 34.87x / 34.70x | 1.000 / 1.000 | **31.82x — VOID** | 3.04e-13 MATCH |
| **512** | **777.959 / 797.021 ms** | **5.899 ms** | **125.19x / 125.45x** | **1.000 / 0.987** | **1.77x** | 8.96e-13 MATCH |

**n=256 is discarded**, not quoted. Its PT spread is **31.82x** — the incumbent varied
by a factor of 32 across its own samples, so its minimum is one escaped sample rather
than a measurement. Both nulls passed and parity matched; neither is sufficient when the
incumbent itself was that unstable. The n=512 row on the same invocation ran at spread
1.77x, which is what a usable incumbent looks like.

The `geqrf` producing `(A, tau)` ran **outside the clock on both arms** — ours before
`Instant::now()`, torch's at `LANES` construction — so this measures `orgqr` and not
`geqrf`'s own 227x defect. Checksum is `|Q|` because each arm forms Q from its own
factorisation and Q is unique only up to column signs.

## The family, now two-thirds measured

| op | role | n=512 standing |
|---|---|---|
| `geqrf` | produce reflectors | **227.58x** |
| `orgqr` | form Q from them | **125.19x** |
| `ormqr` | apply them to a matrix | unmeasured |

Both measured members are catastrophic and both run the same private per-reflector
BLAS-2 loops in `ft-api` — `geqrf_packed_f64` and `apply_reflector_left` — with
identical stride-n indexing on row-major arrays, neither reaching the blocked compact-WY
kernel that `tensor_linalg_qr` already dispatches at `m >= 128 && k >= 16`.

`orgqr` being *less* bad than `geqrf` (125x vs 227x) is consistent with the mechanism:
`orgqr` walks the reflector column `a_packed[i * a_cols + kk]` strided but its second
operand `c[i * ncols + j]` sweeps `j` contiguously in the inner loop, so it gets partial
locality that `geqrf`'s two strided operands do not.

## The ledger figure for this op is refuted for the single-matrix path

The only prior number was **~1.8x** (`torch@8 48 ms vs FT@8 ~86 ms`), cross-run and at
batched-tiny shapes. I flagged it as uninformative rather than a bound when `geqrf` came
in at 226x; that call is now vindicated — the single-matrix path measures **125x**, not
1.8x. The batched regime and the single-matrix regime are different code paths and a
figure from one says nothing about the other.

## Two instrument defects fixed on the way here

**1. An unrunnable binary reported as a passing run.** The rch worker fleet is
heterogeneous in glibc: this host is 2.42, worker `hz2` is ≥2.43. A build landing on
`hz2` links against the newer `libm` and dies at load with
`libm.so.6: version GLIBC_2.43 not found` — while `cargo` returns rc=0. My runner then
printed `VALIDATED_RUN_OK` over it, because it validates result rows and there were
none. **A gate that only inspects rows cannot notice that there are none.** Added GATE 0:
fail on `rc != 0` *or* absence of any `^n=` row, diverting output to a `_failed.log`.
The build loop now verifies with `ldd` that the ELF resolves before trusting it. Since
rch exposes no worker pin, I drained `hz2`, rebuilt onto `vmi1153651`, and re-enabled
`hz2` immediately — blast radius checked first (11/13 healthy, 16/24 slots free, drain
is graceful so no peer job was interrupted).

**2. The PT-spread ceiling finally landed.** I wrote it after an n=1024 SVD row passed
the minimum check at 135.599 ms with a **648x** spread, but the patch was blocked by a
shell guard and I recorded it as owed. It was still missing here, which is why the
n=256 `orgqr` row with a **31.82x** spread was reported as validated instead of voided.
Now in place: reject above 3x, against clean rows that run 1.14–2.48x.

Both defects share the shape that has recurred all session: **the success signal was
derived from something other than the thing that had to succeed.**

## Board

| op | n | standing | null |
|---|---|---|---|
| `geqrf` | 1024 | **457–475x** | 1.000 |
| `geqrf` | 512 | 222–228x | 1.002 |
| **`orgqr`** | **512** | **125.19–125.45x** | **1.000 / 0.987** |
| `geqrf` | 256 | 48x | 0.989 |
| `orgqr` | 256 | *void — PT spread 31.82x* | — |
| `eigh` | 1024 | 8.16–8.23x | 0.994 |
| `eigh` / `qr` / `eigvalsh` | 512 | 5.60x / 5.50x / 4.18x | all PASS |
| SVD | 1024 / 512 | 3.10x / 2.40x | PASS |

`ormqr` is the one remaining unmeasured family member, and the same
`apply_reflector_left`/`apply_reflector_right` sit under it.
