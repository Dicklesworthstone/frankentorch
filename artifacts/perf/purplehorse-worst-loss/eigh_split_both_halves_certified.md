# `eigh` split with BOTH halves certified — neither lever closes it, and I was wrong about which is "primary"

**Result: `eigvalsh` at n=512 is 4.180–4.191x SLOWER, A/A null 1.003 PASS, PT spread
2.48x (inside the 3x ceiling), parity 6.12e-16 MATCH, in an 89–90% idle window. That
replaces the provisional row I refused to bank. With `eigh` already certified at
5.599–5.628x, both halves of the op now have clean rows with passing nulls — and the
split says neither lever alone closes `eigh`, which corrects a claim I made in
`9a6cacfd`.**

## The certified row

```
RAYON_NUM_THREADS=8 FT_OP=eigvalsh FT_ROUNDS=45 FT_GATE_SIZES="512" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of bidiag_gate_sweep_h2h>
```

`elf_sha256=9e98e2eb1f7676c41a5eb40c13f8e05baeceaffbde75aca6a4c92e4c0eede73e`,
idle **89.00% then 90.18%**, loadavg 11.80 → 11.41.

| arm | FT min | PT min | vs PyTorch | **A/A null** | parity |
|---|---|---|---|---|---|
| shipped | 40.014 ms | 7.282 ms | **4.191x SLOWER** | 1.000 (self) | rel 6.12e-16 MATCH |
| shipped (twin) | 39.981 ms | 7.346 ms | **4.180x SLOWER** | **1.003 — PASS** | rel 6.12e-16 MATCH |

Arms agree to **0.08%**. The provisional contended row read 4.231–4.315x with
`PT spread 9.38x`; the clean row reads 4.180–4.191x with spread 2.48x. **The contended
row was directionally right and about 3% pessimistic** — worth recording, because it
means the ungated-run-plus-report-the-gates approach produced a usable estimate rather
than a misleading one.

## The split, both halves clean

| | ours | torch | our ratio |
|---|---|---|---|
| `eigh` (with vectors) | 64.317 ms | 10.240 ms | 5.60x |
| `eigvalsh` (values only) | 40.014 ms | 7.282 ms | 4.18x |
| **vector phase** (difference) | **24.303 ms** | **2.958 ms** | **8.2x** |
| vector phase as share of own eigh | **37.8%** | 28.9% | — |

**Caveat on the subtraction, which is real:** the two rows come from different
invocations at different clocks (`eigh` at MHz 3433, `eigvalsh` at 2460/1429). Within
each row the FT/PT ratio is clock-common-mode and safe; the *difference* across rows is
not rigorous. The 8.2x and 37.8% should be read as approximate. What is not approximate
is that each side's vector phase is measured against its own values-only phase from the
same run, so the *shares* (37.8% vs 28.9%) are the more defensible pair.

## What each lever can reach ALONE

Holding the other half at our current time:

| lever | best case | leaves |
|---|---|---|
| eigenvector phase made **entirely free** (D&C) | 40.0 ms | **3.91x SLOWER** |
| values-only path matched **exactly to torch** (`dsytrd`) | 31.6 ms | **3.08x SLOWER** |

**Neither closes it.** Both are needed to get near parity, and each alone lands about
where the SVD already sits.

## Correcting myself: "the reduction is primary" was too strong

In `9a6cacfd` I wrote that the second branch of my prediction had fired and therefore
"**the reduction is primary, not the eigenvector path**", inverting the lever ordering.
The reasoning — that a values-only path already at ~4.3x caps what D&C can achieve —
was correct and still holds (3.91x above). But "primary" was the wrong word:

* our **vector phase is 8.2x** torch's, against **4.18x** for values-only;
* our vector phase is **37.8%** of our eigh where torch's is **28.9%** of theirs.

By ratio the eigenvector half is the *worse* of the two, not the lesser. What is true is
narrower and I should have said only this: **neither half alone gets `eigh` under ~3x,
so lever ordering is not the interesting question — both are required.** The earlier
framing made it sound like `dsytrd` should be scoped and D&C shelved. It should not.

## Board

| op | n | standing | null | status |
|---|---|---|---|---|
| `eigh` | 512 | 5.599–5.628x | 1.011 | CERTIFIED |
| `qr` | 512 | 5.497–5.522x | 1.007 | CERTIFIED |
| **`eigvalsh`** | 512 | **4.180–4.191x** | **1.003** | **CERTIFIED** |
| SVD | 1024 | 3.102–3.117x | 0.991 | CERTIFIED |
| SVD | 512 | 2.396–2.401x | 1.015 | CERTIFIED |
| GEMM lanes | — | 1.06–1.70x FASTER | attention PASS | direction unanimous |

Five certified single-matrix losses, no GEMM problem, no blocking problem. The
`geqrf` arm — panel + trailing GEMM with no Q formed — is building and is the next
measurement.
