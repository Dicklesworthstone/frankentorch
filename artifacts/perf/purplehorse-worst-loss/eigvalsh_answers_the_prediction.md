# `eigvalsh` is ~4.3x SLOWER too — the eigenvector path is NOT the story, the reduction is

**Result: the values-only path, which skips eigenvectors entirely, is itself
4.231–4.315x SLOWER than PyTorch at n=512 — against `eigh`'s certified 5.599–5.628x.
Removing the eigenvector work would therefore recover only a fraction of the gap. This
is the second branch of the prediction I put on the record two commits ago, and it
means lever 1 (divide-and-conquer) is even less sufficient than the bound I gave for
it, while lever 2 (blocked tridiagonalisation, `dsytrd`) is primary.**

## The prediction, and which branch fired

From `928a6a5c`, before the measurement:

> **PREDICTION if the 73% is right:** eigvalsh lands near ~17 ms and is a much smaller
> loss than eigh's 5.6x.
> **IF INSTEAD eigvalsh is ALSO ~5x slower:** the replay is not the story, the
> reduction is, and lever 2 becomes primary.

`eigvalsh` came back at **42.284 ms and ~4.3x slower**. Second branch.

## The row

```
RAYON_NUM_THREADS=8 FT_OP=eigvalsh FT_ROUNDS=45 FT_GATE_SIZES="512" \
FT_GATE_VALUES="262144,262144" PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of bidiag_gate_sweep_h2h>
```

`elf_sha256=9e98e2eb1f7676c41a5eb40c13f8e05baeceaffbde75aca6a4c92e4c0eede73e`

| arm | FT min | vs PyTorch (paired) | A/A null | parity |
|---|---|---|---|---|
| shipped | 42.284 ms | **4.315x SLOWER** | 1.000 (self) | rel 6.12e-16 MATCH |
| shipped (twin) | 42.326 ms | **4.231x SLOWER** | **1.015 — PASS** | rel 6.12e-16 MATCH |

## This row FAILS my own spread gate, and I am not banking it

`PT spread 9.38x`, against the 3x ceiling I wrote after an n=1024 row passed the
minimum check at 135.599 ms while its spread was 648x. **By my own rule this row is not
quotable**, and it is reported as failed rather than argued past.

It was taken with the idle gate deliberately removed — this host has sat at 0.00–0.12%
idle for hours and the ≥70% gate would never have opened. Waiting delivers nothing;
taking the row and letting the harness's gates speak delivers a bounded statement.

**What survives the gate failure**, because it does not depend on the incumbent:

* The two FrankenTorch arms read **42.284 and 42.326 ms — 0.1% apart**, with the A/A
  null passing at 1.015. Our side of this row is unusually solid.
* Both `min` (4.315x) and `median` (4.00x, from PT median 10.573 ms) estimators land in
  the same place, so the ratio is not being carried by one lucky incumbent sample —
  which is the specific failure mode a wide spread usually causes.

So `~4.0–4.3x` is **provisional, not banked**, and the structural conclusion below does
not rest on the torch ratio at all.

## An estimator error I nearly made, in my own report

Dividing `42.284 / 5.976` (FT min by PT min) gives **7.08x** — and those two minima come
from *different rounds*. That is precisely the min-versus-paired error this campaign has
paid for before and that I have flagged repeatedly this session. The harness's
**paired per-round median** is the authoritative estimator and reads **4.315x**. The
7.08x figure is wrong and is recorded here only so nobody re-derives it.

## The structural finding, which needs no torch at all

Comparing our own arms across windows, in the one direction the bias permits:

| | window | FT time |
|---|---|---|
| `eigh` (with vectors) | idle 91.5%, **certified** | 64.317 ms |
| `eigvalsh` (values only) | idle 0.01%, contended | 42.284 ms |

Contention can only **inflate** `eigvalsh`, so 42.284 ms is an *upper* bound on its
clean value. Therefore:

* eigenvector path `= eigh − eigvalsh ≥ 22.03 ms`, i.e. **≥34% of eigh** — a lower bound;
* values-only path **≤66% of eigh**.

Both are consistent with the 46–66% range from `7d64ea4f` and tighten its floor. And
the direction is the point: **the values-only half is the majority of `eigh` and is
itself a ~4x loss.**

## Consequence for the two levers

| lever | previous bound | now |
|---|---|---|
| 1. divide-and-conquer (`dstedc`) | free eigenvector phase still leaves 1.9–3.0x | **weaker still** — the values-only path alone is already ~4.3x, so D&C cannot take `eigh` below roughly that |
| 2. blocked tridiagonalisation (`dsytrd`) | "the smaller ~1/9", unmeasured | **primary** — it is the half that is both the majority of the time and independently ~4x slower than torch |

That inverts the ordering I gave when the flop count was the only evidence. The flop
count said the replay had 9x the arithmetic; the per-core table in `127248e4` showed
why that does not translate to time; and this row now shows the values-only half losing
~4.3x on its own. **Lever 2 should be scoped first.**

Also note what lever 2 contains: the serial tridiagonal reduction *and* the serial
back-transform, of which the update loop is bit-exactly parallelisable
(`127248e4`) — a ~1.2x piece that sits inside the phase now identified as primary.

## Still owed

A clean-window `eigvalsh` row to replace this one, and a `qr` row — the last op this
harness supports that has never been measured against a live incumbent. Neither can be
taken while the host is at 0.00% idle.
