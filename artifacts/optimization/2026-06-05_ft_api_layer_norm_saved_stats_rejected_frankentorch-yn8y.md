# ft-api LayerNorm saved-stats pass rejected

Bead: `frankentorch-yn8y`
Date: 2026-06-05
Agent: Codex

## Profile-backed target

Criterion target:

```text
cargo bench -p ft-api --bench ops_bench -- layer_norm --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Clean baseline worktree: `/data/projects/.scratch/frankentorch-yn8y-baseline-20260605T014700`
Baseline commit: `36d0237d`
Worker: `ts2`

```text
layer_norm/nograd_2048x1024 [6.8415 ms 6.9444 ms 7.0749 ms]
layer_norm/grad_2048x1024   [124.04 ms 125.03 ms 126.04 ms]
```

Candidate worktree: `/data/projects/.scratch/frankentorch-yn8y-after-20260605T015830`
Candidate commit: `7203edeb` (includes landed LayerNorm commit `89769828`)
Worker: `ts2`

```text
layer_norm/nograd_2048x1024 [7.3575 ms 7.4886 ms 7.6147 ms]
layer_norm/grad_2048x1024   [119.88 ms 120.51 ms 121.20 ms]
```

## One lever assessed

Commit `89769828` added `layer_norm_forward_with_stats_f64` and
`layer_norm_backward_with_stats_f64`, then changed the f64 affine grad
LayerNorm fast path to save forward means/rstds and reuse them in backward
instead of recomputing those row statistics.

The same commit also included unrelated `ft-kernel-cpu` cleanup hunks in
`conv2d_im2col_f64`, `conv3d_im2col_f64`, `sort_radix_perm`, Winograd helper
loops, and a clippy allow. Those hunks are not part of the LayerNorm lever.
The commit had already landed and been pushed to `main`/`main:master` before
this closeout, so this artifact records the scope issue instead of rewriting
peer-owned history.

## Score

Primary p50 ratio:

```text
125.03 ms / 120.51 ms = 1.04x
```

Opportunity score:

```text
Impact 1.04 * Confidence 0.95 / Effort 1.0 = 0.99
```

Verdict: reject. The pass does not clear the required `Score >= 2.0` gate.

## Isomorphism status

- Ordering preserved: yes. Output rows and affine gradient reductions remain
  row-major.
- Tie-breaking unchanged: N/A. No comparisons or tie rules are involved.
- Floating-point: expected equivalent for the intended LayerNorm lever because
  saved stats are computed by the same forward formula and reused by backward.
  Direct old-vs-new `to_bits()` tests were not present in the landed commit;
  this is a proof gap, not a known mismatch.
- RNG unchanged: yes. No RNG path changed.
- Golden outputs: no new golden fixture was added in the landed commit.

## Additional measurements

The exact `layer_norm` filter is the score source above. Two directional live
after runs on `ts1` showed p50 grad values of `87.372 ms` and `94.763 ms`, but
without a same-worker clean baseline they are not campaign-quality evidence.

An isolated `layer_norm/grad_2048x1024` filter produced a same-worker `ts2`
pair of `260.99 ms -> 250.31 ms`; this also fails the score gate and confirms
the lever is not a major win on the measured workload.

## Next deeper target

Do not keep iterating on saved-stat rematerialization. The next higher-EV
training-path primitive should attack the generic autograd/session overhead:
compact custom saved-state payloads or zero-copy in-place parameter/gradient
access, with the existing `frankentorch-mqt7` optimizer bead as the stronger
profile-backed candidate if `ft-api` ownership is clear.
