# `eigh` n=1024 is 8.2x SLOWER — and our time scales n^2.96 where torch's scales n^2.14

**Result: `eigh` at n=1024 is 8.163–8.227x SLOWER than PyTorch, A/A null 0.994 PASS,
PT spread 1.45x, parity 4.59e-16 MATCH. That is the worst certified loss in the tree.
More importantly, three sizes in one invocation give the scaling exponents: our `eigh`
grows as n^2.96 — cubic, to two decimal places — while torch's grows as n^2.14. The
gap widens as n^0.82 and will keep widening. That is the first *measured* evidence for
the algorithmic story I had only been able to argue from flop counts and source
reading.**

## The rows

```
FT_OP=eigh FT_ROUNDS=25 RAYON_NUM_THREADS=8 FT_GATE_SIZES="256,512,1024" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of bidiag_gate_sweep_h2h>
```

`elf_sha256=9e98e2eb1f7676c41a5eb40c13f8e05baeceaffbde75aca6a4c92e4c0eede73e`,
idle **92.27% then 89.93%**, loadavg 6.73 → 8.22.

| n | FT min (arm0/arm1) | PT min | standing | **A/A null** | PT spread | parity |
|---|---|---|---|---|---|---|
| 256 | 8.298 / 8.061 ms | 2.795 ms | 2.873x / 2.830x | **1.053 — FAIL** | 1.35x | 3.06e-16 MATCH |
| 512 | 65.067 / 64.835 ms | 11.903 ms | 5.520x / 5.355x | **1.009 — PASS** | 1.31x | 6.12e-16 MATCH |
| **1024** | **502.695 / 504.096 ms** | **54.490 ms** | **8.163x / 8.227x** | **0.994 — PASS** | 1.45x | 4.59e-16 MATCH |

n=256's null misses at 1.053 and that row is **not quoted**. n=512 reproduces the
separately certified 5.599–5.628x to within 3%, on a different day and a different
window, which is the cross-check that makes the n=1024 row credible.

## The scaling, which is the real finding

| interval | ours | torch |
|---|---|---|
| 256 → 512 | **n^2.97** | n^2.09 |
| 512 → 1024 | **n^2.95** | n^2.19 |
| **overall** | **n^2.96** | **n^2.14** |

Our `eigh` is cubic to two decimal places. Torch's is barely above quadratic. **The gap
grows as n^0.82**, so it is not a constant-factor problem and no amount of constant-factor
tuning reaches it.

This is exactly the signature the algorithm mismatch predicts and it now has numbers
behind it rather than inference:

* ours accumulates ~2n² Givens rotations across all n rows — **strictly O(n³)**, and
  measured at n^2.96;
* LAPACK `dsyevd` uses divide-and-conquer (`dstedc`), which is **subcubic in practice** —
  and torch measures n^2.14, right where that predicts.

Every earlier argument for this was indirect: a flop count (`928a6a5c`), a source
comment conceding "~11x slower than LAPACK syevd", the absence of any `dstedc` in the
tree. Two of my causal stories this session were refuted after looking equally
plausible. **This one is measured, across three sizes, with passing nulls at the two
that matter.**

## What it does to the lever bounds

The n=512 bound in `fb4cd364` — vectors made free leaves 3.91x — is **size-specific and
gets better with n**, because the cubic term is in the vector phase. It should not be
quoted at n=1024 without re-measuring `eigvalsh` there.

Conversely the case for divide-and-conquer strengthens sharply: it is the only lever
that changes the *exponent*. Blocked tridiagonalisation (`dsytrd`), the bit-exact
back-transform parallelisation (~1.2x), AVX2 (~1.07x) and dispatch tuning (<5%) are all
constant-factor levers against a term growing as n^0.82.

**Extrapolation, explicitly NOT measured:** the same exponents put n=2048 at ~14.5x.
Recorded as an extrapolation only — three points is thin, and I have already been
burned this session by a tidy model that one check refuted.

## Board — new worst

| op | n | standing | null | status |
|---|---|---|---|---|
| **`eigh`** | **1024** | **8.163–8.227x** | **0.994** | **CERTIFIED — worst in tree** |
| `eigh` | 512 | 5.599–5.628x | 1.011 | CERTIFIED |
| `qr` | 512 | 5.497–5.522x | 1.007 | CERTIFIED |
| `eigvalsh` | 512 | 4.180–4.191x | 1.003 | CERTIFIED |
| SVD | 1024 | 3.102–3.117x | 0.991 | CERTIFIED |
| SVD | 512 | 2.396–2.401x | 1.015 | CERTIFIED |
| GEMM lanes | — | 1.06–1.70x FASTER | attention PASS | direction unanimous |

Note the SVD at n=1024 is 3.10x while `eigh` at the same size is 8.2x — the two
decompositions diverge sharply with n, which is further evidence against any single
shared constant-factor cause and consistent with the SVD's bidiagonal QR replay being
better behaved than eigh's QL accumulation.

## Caveats

* `FT_ROUNDS=25` here rather than 45, because one n=1024 `eigh` is ~0.5 s. The nulls are
  correspondingly looser, and n=256's failed.
* My incumbent-plausibility gate still has **no banked `eigh` reference** and compared
  against the SVD's ~122.604 ms at n=1024, passing vacuously. Torch's figures are
  credible on their own terms — 2.795 / 11.903 / 54.490 ms scales as n^2.14, which is
  the right shape for `dsyevd` — but the gate did not earn its pass. Per-op references
  remain owed.
* Core clock varied across sizes within the run (1429 MHz at n=256, 3433 at n=512/1024)
  as the cores boosted under load. Both arms share the clock at each size, so the
  per-size ratios are common-mode; the *absolute* n=256 numbers are not comparable with
  the larger two.
