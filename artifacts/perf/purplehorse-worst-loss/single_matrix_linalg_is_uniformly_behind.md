# Every single-matrix dense decomposition loses 2.4–5.6x — one root cause, not three

**Result: `qr` at n=512 is 5.497–5.522x SLOWER than PyTorch, A/A null 1.007 PASS,
PT spread 1.70x, parity 3.85e-13 MATCH, in a 90.5%-idle window. That is essentially
tied with `eigh`'s certified 5.6x. With the SVD at 2.40x (n=512) / 3.10x (n=1024) and
`eigvalsh` at ~4.3x, every single-matrix dense decomposition this harness can now reach
is behind by 2.4–5.6x — while the batched-tiny regime wins 4–10x. That is one shared
root cause, and it reframes "eigh is the worst op" as the wrong unit of analysis.**

## The `qr` row

```
RAYON_NUM_THREADS=8 FT_OP=qr FT_ROUNDS=45 FT_GATE_SIZES="512" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of bidiag_gate_sweep_h2h>
```

`elf_sha256=9e98e2eb1f7676c41a5eb40c13f8e05baeceaffbde75aca6a4c92e4c0eede73e`,
idle **90.54% then 91.30%**, loadavg 13.02 → 12.38, iowait 134.

| arm | FT min | PT min | vs PyTorch | **A/A null** | parity |
|---|---|---|---|---|---|
| shipped | 40.707 ms | 6.912 ms | **5.522x SLOWER** | 1.000 (self) | rel 3.85e-13 MATCH |
| shipped (twin) | 40.601 ms | 6.441 ms | **5.497x SLOWER** | **1.007 — PASS** | rel 3.85e-13 MATCH |

Null passes, spread 1.70x is inside the 3x ceiling, arms agree to 0.3%, parity matches.
Both arms return `|diag(R)|` — R's diagonal is sign-ambiguous between implementations
while its magnitudes are determined, which is what makes that column meaningful.

## The pattern, now that the family is covered

| op | n | standing | null | status |
|---|---|---|---|---|
| **`eigh`** | 512 | **5.599–5.628x SLOWER** | 1.011 | CERTIFIED |
| **`qr`** | 512 | **5.497–5.522x SLOWER** | 1.007 | certified-quality |
| `eigvalsh` | 512 | ~4.23–4.32x SLOWER | 1.015 | provisional (PT spread 9.38x) |
| SVD | 1024 | 3.102–3.117x SLOWER | 0.991 | CERTIFIED |
| SVD | 512 | 2.396–2.401x SLOWER | 1.015 | CERTIFIED |

**Nothing in the single-matrix family is at parity.** The spread across four distinct
algorithms — Householder tridiagonalisation + QL, Householder QR, Golub-Reinsch
bidiagonalisation — is only 2.3x, which is a much narrower band than the algorithms
themselves differ by. That is the signature of a shared cause rather than three
independent implementation defects.

## The shared cause, and why the batched wins do not contradict it

For a **single large matrix**, LAPACK/MKL runs *blocked, BLAS-3* factorisations:
`dgeqrf` (panel factorisation + trailing update via GEMM), `dsytrd`, `dgesdd`. The
trailing update is a matrix-matrix product and reaches a large fraction of peak.

Ours are unblocked or partially blocked and, where the source is explicit, **serial
scalar**: the eigh reduction's own PERF NOTE says *"serial scalar … ~11x slower than
LAPACK syevd"*, and the per-core table in `127248e4` shows 2.1e9 flops running on one
core of a 64-core machine at n=1024.

For **batched tiny matrices** the advantage inverts and we win 4–10x — but for an
orthogonal reason that this measurement leaves untouched: torch's CPU batched
factorisation *loops serially over the batch* (structurally proven — torch
`svdvals [2000,64]` reads 262/299/330 ms at 1/8/32 threads, **slower** with more cores),
so we win by parallelising over planes. Nothing about that helps a single matrix, where
there is only one plane and the win must come from within it.

So both facts hold simultaneously and neither is evidence about the other:

* batched-tiny: **we win 4–10x**, by parallelism across planes;
* single-large: **we lose 2.4–5.6x**, for want of blocked BLAS-3 within the plane.

## What this does to the lever list

The per-op levers I bounded earlier were all real and all insufficient — and now it is
clear why. They were tuning constants inside unblocked algorithms:

| lever | bound | verdict |
|---|---|---|
| SVD expansion phase | removing **all** of it leaves 1.734x | insufficient |
| SVD reduction parallel dispatch | **<5%** | insufficient |
| SVD `x86-64-v3` AVX2 | ~1.07x | insufficient |
| SVD `form_p` gate | measured **no-op** | none |
| eigh divide-and-conquer | leaves ≥ ~4.3x (values-only floor) | insufficient |
| eigh back-transform update loop | ~1.2x, bit-exact | insufficient |

**The single lever that addresses all five ops at once is blocked BLAS-3
factorisation** — panel + trailing GEMM — which is the same family that already won
blocked-Cholesky and blocked-QR-panel in this tree, and which the eigh source names
directly as *"the real lever … a multi-turn rewrite"*. `gemm::dgemm` already exists and
is already the fast path elsewhere.

This is a multi-session programme, not a lever. Stating it as such is the useful output:
the per-op micro-levers have now been measured to exhaustion and none of them closes
anything, which is itself the argument for the rewrite.

## Provenance caveats I am not hiding

* My incumbent-plausibility gate still has **no banked `qr` or `eigh` reference** and
  passes vacuously for both, comparing against the SVD's ~32.075 ms. The `qr` incumbent
  is credible independently — 6.4–6.9 ms is the right order for LAPACK `dgeqrf` at
  n=512 on 8 threads, and it reproduced across the two arms — but the gate did not earn
  its pass. Per-op references are owed.
* `MHz 1429/1429` on this row: the cores were at the frequency floor throughout. Both
  arms were, so the comparison is common-mode, but the absolute times are not
  comparable with rows taken at 3.4 GHz.
* `eigvalsh` remains provisional pending a clean-window re-take.
