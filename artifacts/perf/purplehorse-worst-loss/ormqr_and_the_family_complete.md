# `ormqr` completes the Householder family — ~253x at n=512, ~473x at n=1024, and my own gate refuses to bank it

**Result: all three LAPACK Householder primitives now have live-torch numbers.
`ormqr` measures 251.6–259.2x SLOWER at n=512 (four arm-readings, two independent
windows) and 472.7–485.2x at n=1024. Every A/A null passes at 1.000–1.005 and parity
matches at 7.2e-13–1.5e-12, but **every row is voided by my own PT-spread gate**,
because torch's `ormqr` is the shortest incumbent in the family and its sampling
scatter never settles under the 3x ceiling. I am reporting it as reproduced-but-unbanked
rather than arguing past the rule.**

## The rows

```
FT_OP=ormqr RAYON_NUM_THREADS=8 FT_GATE_VALUES="262144,262144" \
PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python <snapshot of bidiag_gate_sweep_h2h>
```

`elf_sha256` snapshotted as `bidiag_elf_ormqr` (built on `vmi1156319`, `ldd` clean).

| n | window | FT min | PT min | ratio | A/A null | PT spread | parity |
|---|---|---|---|---|---|---|---|
| 512 | #1 | 786.459 / 795.614 ms | 2.687 ms | 254.83x / 251.56x | 1.000 / 1.000 | **3.54x VOID** | 7.20e-13 |
| 512 | #2 | 796.739 / 793.653 ms | 2.742 ms | 259.17x / 252.83x | 1.000 / 1.005 | **20.86x VOID** | 7.20e-13 |
| 1024 | #3 | 7318.337 / 7272.715 ms | 14.299 ms | 472.68x / 485.16x | 1.000 / 1.003 | **3.56x VOID** | 1.54e-12 |

**7.3 seconds** for a single 1024×1024 `ormqr` against torch's 14.3 ms.

### Why every row is void, and why I am not overriding it

GATE 2b rejects a row whose incumbent spread exceeds 3x, because a wide spread means the
PT *minimum* is one sample that escaped contention rather than a measurement. That rule
caught a 648x-spread SVD row and a 31.82x `orgqr` row that all other gates passed. It is
the correct rule and I am keeping it.

What argues the number is nonetheless real — recorded so the next session need not
re-derive it:

* **Our arm is stable to 1.3%** across all six readings (786.459, 795.614, 796.739,
  793.653 ms at n=512; 7318.337, 7272.715 at n=1024).
* **Every A/A null passes**, 1.000–1.005.
* **The ratio reproduces across three independent windows** — 251.6x, 252.8x, 254.8x,
  259.2x at n=512, a 3% band.
* **The incumbent is physically plausible**: torch `ormqr` at n=512 is 2.687 ms and at
  n=1024 is 14.299 ms — 5.3x for a 2x size step, i.e. sub-cubic, which is what a blocked
  implementation does and is the same shape torch shows on `geqrf` (n^1.84–2.10).

Four agreeing measurements across three windows is a different epistemic position from
one escaped minimum. But "different" is not "passes", so this stays unbanked.

**The gate is arguably too strict for this specific op, and that is a finding about the
instrument, not a licence.** torch's `ormqr` is the shortest incumbent in the family
(2.687 ms against `geqrf`'s 2.201 ms at spread 1.78x, `orgqr`'s 5.899 ms at 1.77x), yet
it scatters where they do not — so the scatter is a property of torch's `ormqr`, not of
the window. Raising the ceiling for one op to make a number quotable is exactly the
move this campaign has refused elsewhere; the honest fix is a longer op, and n=1024
already tried that and still read 3.56x.

## A prediction of mine this refutes

Last cycle I predicted `ormqr` would land **between** `orgqr` (125x) and `geqrf` (227x),
reasoning that it shares `apply_reflector_left` with `orgqr` and that operand locality
would place it in that band. It lands **above both**, and the reason is not on our side
at all:

| | ours | torch | ratio |
|---|---|---|---|
| `orgqr` n=512 | 777.959 ms | 5.899 ms | 125.2x |
| `ormqr` n=512 | 786.459 ms | 2.687 ms | ~253x |

**Our two ops cost the same — 1.1% apart. Torch's differ by 2.2x.** The entire gap
between 125x and 253x is torch's `ormqr` being cheaper than its `orgqr`, not our
`ormqr` being slower than our `orgqr`.

That sharpens the defect rather than softening it: **our `orgqr` and `ormqr` are
indistinguishable in cost because the naive per-reflector loop ignores the structural
difference that makes them different operations.** Applying reflectors to a general C is
genuinely less work than accumulating Q from the identity, and torch exploits that while
we do not. The spread of ratios across the family (125x / 227x / 253x) says more about
which torch kernel is efficient than about which of our loops is worst — all three of
ours are the same bad loop.

## The family, complete

| op | role | ours n=512 | torch n=512 | ratio | status |
|---|---|---|---|---|---|
| `geqrf` | produce reflectors | 559.481 ms | 2.201 ms | **227.6x** | CERTIFIED |
| `orgqr` | form Q | 777.959 ms | 5.899 ms | **125.2x** | CERTIFIED |
| `ormqr` | apply them | 786.459 ms | 2.687 ms | ~253x | reproduced, gate-void |
| `qr` | all of the above, blocked | **40.707 ms** | 6.441 ms | **5.50x** | CERTIFIED |

All three primitives are private per-reflector BLAS-2 loops in `ft-api`
(`geqrf_packed_f64`, `apply_reflector_left`, `apply_reflector_right`) that never reach
the blocked compact-WY kernel `tensor_linalg_qr` dispatches at `m >= 128 && k >= 16`.

**Our own `qr` is 32.9x faster than our own `geqrf` + `orgqr`**, computing the same
factorisation — and `qr` does strictly more work, since it also forms Q.

## Sizing the family fix

At n=512 the three naive primitives cost 559 + 778 + 786 = **2124 ms** of work that the
blocked path does in a fraction of that. The re-route is one bead
(`frankentorch-geqrf-misses-blocked-kernel-1zp6r`) whose open questions are unchanged
and still unassumed: the packed-V layout our own consumers expect, and bit-exactness
under the ratified tolerance policy, since the blocked path is documented as not
bit-identical above its threshold.
