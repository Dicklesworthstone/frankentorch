# uufyp, re-measured with conditioning: the backward stage is ~5.6 ms / ~18x, not 11 ms / 34x

`27aci` showed the original decomposition was measuring allocator history. This is the
same decomposition with `condition_allocator()` before every timed rep and an A/A null
lane added. **The conclusion survives; the number was inflated ~2x.**

## Result (three runs, conditioned)

| stage | run 1 | run 2 | run 3 | vs its kernel |
|---|---|---|---|---|
| `session_new` | 0.001 | 0.000 | 0.001 | free |
| `leaf_build` | 0.828 | 0.976 | 0.783 | of which bare clone 0.659 / 0.597 / 0.826 |
| `forward` | 1.234 | 1.048 | 1.440 | raw_fwd ~0.7–0.9 |
| **`backward`** | **5.620** | **6.024** | **5.063** | **raw_bwd ~0.32 → ~18x** |
| `sum` | 0.510 | 0.618 | 1.243 | — |
| `grad_fetch_sum` | 1.102 | 0.769 | 1.255 | — |
| *(session total)* | 9.294 | 9.434 | 9.785 | |

**A/A null on the through-sum lane: 1.5% / 13.7% / 13.2%.** Compare the unconditioned
harness, where the analogous check was 83% and the bare-clone control read 164–234% of
the stage containing it. The conditioning fixed a real pathology: the clone ratio is
now 61–105%, i.e. physical.

## Corrected vs withdrawn

| | unconditioned (81c9d4ad) | conditioned (this) |
|---|---|---|
| session total | 16.3–17.1 ms | **9.3–9.8 ms** |
| backward stage | 10.9–11.8 ms | **5.1–6.0 ms** |
| backward / kernel | ~34x | **~18x** |
| backward share of session | ~65–70% | **~55–60%** |

**Roughly half of the original 11 ms was allocator first-touch, not engine work.** The
34x is withdrawn and replaced with ~18x.

## What still holds, and it is the part that mattered

- `tensor_backward` is **still the dominant stage** — 5.6 ms of a 9.5 ms session, more
  than the other five stages combined.
- It is **still ~18x the kernel it dispatches** (0.32 ms), which is the fact that
  redirected `k1h8g` away from `ft-kernel-cpu`. An 18x gap redirects exactly as well as
  a 34x one.
- `session_new` is free (0.001 ms) and `leaf_build` is essentially the input clone —
  both were already same-size comparisons and both are unchanged.

So `uufyp`'s conclusion is restored on a defensible basis. Nothing downstream of it
needs revisiting: `k1h8g`'s redirect stands, and no lever was built on the 34x.

## Caveat

**Host load was 82–143 during these runs** — extremely busy, far worse than the
unconditioned runs it replaces. That the three agree within ~16% on the backward stage,
with A/A nulls of 1.5–13.7%, is itself evidence the conditioning made the measurement
robust rather than merely smaller. But the absolute milliseconds should be re-taken
quiet before being quoted anywhere load-sensitive; the **ratio** (~18x) is the durable
claim.

Still probe-framed: fresh session, leaf clone and gradient sum inside the timed region,
so these are not comparable to h2h op-work figures. Single lane, single shape, one
host. No PyTorch arm.

## What remains open

Where the 5.6 ms goes *inside* the backward pass is still unattributed — the candidate
list in `27aci` (create_graph dispatch, gradient accumulation, the pad/unpad pair,
full-size per-backward inits, tape retention) is untouched by this re-measure. That is
the next step, and it now starts from a trustworthy baseline.
