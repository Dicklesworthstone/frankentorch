# ft-nn MHA Batched Heads Pass 27

- Bead: `frankentorch-xly2`
- Skills: `/profiling-software-performance`, `/extreme-software-optimization`, `/alien-graveyard`, `/alien-artifact-coding`
- Crate: `ft-nn`
- Target benchmark: `multihead_attention/forward_8x64x128_h8`

## Profile Target

Fresh rch Criterion profiling kept `ft-nn::MultiheadAttention::forward_qkv`
as the active residual hotspot after earlier scale-tensor reuse. The current
implementation still ran one separate narrow/scale/BMM/softmax/BMM chain per
head, then concatenated the head outputs.

Baseline:

```text
worker: vmi1227854
command: rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_8x64x128_h8 --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [40.116 ms 52.915 ms 69.258 ms]
outliers: 1 high severe
```

## Alien Recommendation Card

Change: reshape projected Q/K/V into `[batch * heads, seq, head_dim]`, process
all heads as one batched attention workload, then restore `[batch, seq, embed]`.

Hotspot evidence: Criterion p50 `52.915 ms` for `forward_8x64x128_h8`.

Mapped graveyard sections:

- `alien_cs_graveyard.md` §8.2: vectorized execution and morsel-driven
  parallelism, specifically amortizing operator overhead over cache-sized
  batches and unlocking bulk execution.
- `high_level_summary...md` FrankenSuite matrix: vectorized execution processes
  data in cache-sized batches as a high-impact throughput primitive.
- `alien_cs_graveyard.md` appendix warning: constants and cache behavior can
  defeat elegant primitives, so the after benchmark is the acceptance gate.

EV score: Impact 3 * Confidence 2 * Reuse 3 / Effort 1 / AdoptionFriction 1 = 18.0.

Priority tier: A. This is a direct graph-level batching lever inside the current
MHA API, not a whole-stack numerical rewrite.

Adoption wedge: preserve the public `MultiheadAttention` API and use only
existing autograd-aware tensor operations.

Budgeted mode: deterministic single pass over the existing tensor graph; no
runtime search, scheduler, or unbounded allocation loop. On shape/API failure,
the existing tensor operation errors propagate.

Expected-loss model: action `batch_heads` minimizes measured per-head graph
overhead; fallback action `per_head_loop` is the manual revert if golden output
or Criterion score fails.

Fallback trigger: revert the lever if the pass-local golden fixture changes, an
MHA backward test fails, or p50 improvement scores below 2.0.

## Alien Artifact Proof

Selected family: certified rewrite pipeline for a numerical graph, with
row-major layout isomorphism as the runtime artifact.

Assumptions:

- `tensor_reshape` and `tensor_permute` materialize row-major values
  deterministically through existing autograd-aware operators.
- `tensor_bmm` computes each independent batch item with the same inner dot
  order as the prior per-head `tensor_bmm`.
- `tensor_softmax(scores, 2)` normalizes each `[S_q, S_k]` row independently and
  does not couple rows across the flattened batch-head dimension.

Proof obligations:

- Ordering: old head loop order `(batch, head, seq, feature)` is transformed to
  `[batch * heads, seq, feature]`, then back through `[batch, head, seq,
  feature] -> [batch, seq, head, feature] -> [batch, seq, embed]`.
- Tie-breaking: no comparisons or tie-breakers are introduced outside existing
  softmax internals.
- Floating point: per-head scale, BMM, softmax, and weighted-value BMM
  arithmetic remains inside the same existing kernels; rows are independent and
  no cross-head reduction is introduced.
- RNG: MHA forward uses no RNG.
- Errors: the same tensor operations provide shape and dispatch errors; no new
  error class is introduced.
- Golden output: `ft_nn_mha_batched_heads_pass24.txt` has sha256
  `d3b78b1fbb9f1cbd157bd420c28f40d678f5e9b75ab27e4a30a23fc47ecc03b5`.

## One Lever

Replace the per-head loop with a batched-head path:

1. reshape Q/K/V to `[batch, seq, heads, head_dim]`;
2. permute to `[batch, heads, seq, head_dim]`;
3. reshape to `[batch * heads, seq, head_dim]`;
4. run one scale, BMM, softmax, and BMM sequence;
5. reshape and permute back to `[batch, seq, embed]`.

No kernel implementation, public API, parameter initialization, or optimizer
behavior changed.

## Result

After:

```text
worker: vmi1293453
command: rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_8x64x128_h8 --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [15.341 ms 15.739 ms 16.169 ms]
```

Delta:

- p50: `52.915 ms -> 15.739 ms`
- improvement: about 70.3 percent faster, 3.36x by p50
- confidence: capped for cross-worker comparison, but delta is far above shared-host Criterion noise
- score: Impact 3 * Confidence 2 / Effort 1 = 6.0
- decision: keep

## Gates

- `rch exec -- cargo test -p ft-nn mha_scale_reuse_golden_output_matches_fixture -- --nocapture` passed after the lever.
- `rch exec -- cargo test -p ft-nn mha_batched_heads_golden_output_matches_fixture -- --nocapture` passed.
- `rch exec -- cargo test -p ft-nn mha -- --nocapture` passed: 7/7 MHA tests, including backward gradients.
- `rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_8x64x128_h8 --warm-up-time 1 --measurement-time 5 --sample-size 20` passed before and after.
- Closeout confirmation benchmark on worker `vmi1149989`: `[18.123 ms 19.576 ms 21.615 ms]`.
- `rch exec -- cargo check -p ft-nn --all-targets` passed; RCH later reported a target-artifact retrieval warning, but the remote command finished with exit 0.
- `rch exec -- cargo clippy -p ft-nn --all-targets --no-deps -- -D warnings` passed.
- `rch exec -- cargo fmt -p ft-nn --check` passed after formatting the long `session.full` line.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed.
- `git diff --check` passed.
- UBS on changed source/evidence files returned nonzero on existing `ft-nn/src/lib.rs` inventory: 99 equality-comparison false positives, 6730 warnings, and 459 info items. UBS reported its own formatting, clippy, cargo check, tests-build, cargo-audit, and cargo-deny sections clean; reported critical locations are pre-existing equality checks outside the new MHA batched-head rewrite.
