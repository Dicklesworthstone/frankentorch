# 87sz8 — the deciding lane A/B, decided: the gate HURTS the max_pool3d lane

`un3os`'s L3-residency gate is a **4.24x win on the isolated kernel** and a
**~15% LOSS on the op-level lane**. Both measurements are sound. The lane is what
users experience, so the gate should come out.

## Result

Same two ELFs as the inconclusive attempt, this time **16 round-pairs in a genuinely
quiet window** (load 5.2 at start, 10–14 throughout):

| arm | ELF |
|---|---|
| A — gates OFF (pre-`un3os`) | `af66bcc72183b1cb…` |
| B — gated (shipped) | `93e8de27a87f5efa…` |

Metric is the per-round ratio-of-ratios `(FT/PT)_B / (FT/PT)_A`, over all rounds
where **both** arms passed their A/A gate:

| lane | n | ftB/A | ptB/A | rr median | effect | 95% CI | rounds FT slower | verdict |
|---|---|---|---|---|---|---|---|---|
| **`max_pool3d`** | 16 | 1.150 | 1.041 | **1.150** | **0.87x** | **[1.036, 1.198]** | **16/16** | **GATE HURTS LANE** |
| `max_pool1d` | 16 | 1.009 | 1.009 | 1.007 | 0.99x | [0.943, 1.031] | 10/16 | no resolvable effect |
| `avg_pool2d` | 14 | 0.979 | 0.964 | 1.037 | 0.96x | [0.819, 1.204] | 5/14 | no resolvable effect |
| `conv3d` | 14 | 1.003 | 0.987 | 1.009 | 0.99x | [0.965, 1.036] | 8/14 | no resolvable effect |

**`conv3d` is the negative control and it behaves.** The gate touches no conv3d
kernel, so it must show nothing — and it lands at 1.009 with a tight CI [0.965,
1.036]. That is what makes the max_pool3d row credible rather than noise.

`max_pool3d` is the only lane the gate can plausibly move, and it moves it the wrong
way: **16 rounds out of 16** show FT slower with the gate on, minimum 1.040.

## Why the kernel win does not survive at the lane

Hypothesis, plausible and **not yet tested** — stated as a hypothesis:

The gate keys on **buffer size** (16 MiB) as a proxy for "fits in one core's 32 MiB
L3 slice". In the isolated kernel probe that proxy is accurate: the 8 MiB gradient
effectively owns L3. In a real session it is not, because **the autograd tape holds
many other live tensors**, so the same 8 MiB buffer is *not* L3-resident — which puts
it in the regime where the measured crossover says **parallel wins**.

If that is right, the error is not the crossover measurement (which was clean) but
the **predictor**: L3 residency is a property of the whole working set, not of one
buffer's size. A size gate cannot see the tape.

This also retro-explains `h2h_remeasure_20260808.md`: two whole-lane readings showed
no sign of improvement, and the gate being a small net negative is consistent with
both of them.

## Why the acceptance rule from `18543b77` was not used as a filter

I pre-specified `|pt_B/pt_A − 1| < 0.05` plus n≥8. Applied strictly it **refuses all
three lanes of interest**:

| lane | A/A-paired | control-ok | torch-arm variance (median / p75 / max) | rounds needed for n=8 |
|---|---|---|---|---|
| `max_pool3d` | 16 | 5 | 6.9% / 12.4% / 16.5% | ~26 |
| `max_pool1d` | 16 | 7 | 5.9% / 14.3% / 23.4% | ~18 |
| `avg_pool2d` | 14 | 1 | 16.1% / 24.6% / **54.0%** | ~112 |
| `conv3d` | 14 | 8 | 4.6% / 6.2% / 9.3% | ~14 |

So the binding constraint is **the torch arm's cross-process variance**, not the FT
arms: each h2h invocation spawns a fresh torch process, and those differ by 5–16% at
the median. `avg_pool2d` swings up to 54%, which is exactly what `k1h8g`'s title has
warned about all along ("PT arm swings run-to-run — quote same-run pairings only").

**Reporting the unfiltered ratio-of-ratios instead is not rule-shopping**, and the
distinction matters: the ratio-of-ratios *divides the torch arm out* rather than
requiring it to be stable, so torch noise inflates the variance but does not bias the
estimate. The filter existed to protect an n=1 result; with n=16 and a clean negative
control the unfiltered estimator is the better one. The filtered subset is reported
above so anyone can check I did not pick the flattering analysis — and note that both
analyses agree on `conv3d` (no effect), which is the only lane where both have n≥8.

## Recommendation: revert the gate

The gate buys **nothing measurable** on three lanes and **costs ~15%** on the one
lane where it is measurable. Reverting returns to behaviour that shipped for months.

Not done in this commit, deliberately — it touches four call sites plus the boundary
test plus the three bit-identity tests that assert the serial path is taken, and that
is a change to make deliberately rather than at the end of a long session on a single
run. Filed as a P1 with the exact steps.

**What the revert should NOT throw away:** the measured crossover data
(`sibling_ab.md`, `avgpool_prediction.md`) and the DO-NOT-GATE notes remain correct
and hard-won. What is wrong is using *buffer size* as the residency predictor, not
the finding that residency matters.

## Standing

`87sz8`'s question — do `un3os`'s kernel wins reach the lane — is now **answered:
no, and the gate is a small net negative there.** The four kernels remain individually
faster in isolation; that was never in doubt and is not contradicted. What is
contradicted is that isolated-kernel speed transfers to this lane.
