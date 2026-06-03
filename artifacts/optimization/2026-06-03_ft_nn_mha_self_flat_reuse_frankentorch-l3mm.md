# ft-nn MHA self-attention flat reshape reuse

Bead: `frankentorch-l3mm`

## Target

Profile-backed fallback target after `br ready --json` stayed empty and active
perf beads were already claimed on `ft-kernel-cpu` and `ft-optim`.

Baseline:

```text
worker: vmi1293453
command: rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_8x64x128_h8 --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [17.030 ms 17.976 ms 19.277 ms]
outliers: 1 high mild
```

Hotspot: `MultiheadAttention::forward_qkv` reshaped the same self-attention
input to `[N * S, E]` three times before Q/K/V projection.

## Alien Recommendation Card

Primitive: constants-kill-you graph-node elimination. Remove repeated
materialization of an identical tensor layout in the hot self-attention path
while keeping all numeric kernels unchanged.

EV: Impact 2 * Confidence 3 * Reuse 3 / Effort 1 / Friction 1 = 18.

Fallback: revert the hunk if golden output plus gradients drift, any focused MHA
test fails, or Criterion after-run scores below 2.0.

## One Lever

When `query == key && key == value`, reuse the first flat reshape node as the
projection input for K and V. Cross-attention keeps the old independent reshape
path.

No public API, parameters, initializers, matmul kernels, softmax kernels,
dispatch paths, or optimizer behavior changed.

## Isomorphism Proof

- Ordering: forward tensor element order is unchanged; Q/K/V projections still
  run in the same Q then K then V order.
- Tie-breaking: no comparisons or tie-breakers were added.
- Floating point: per-output projection, scale, BMM, softmax, weighted BMM, and
  output projection arithmetic stays inside the same existing kernels; the pass
  only removes duplicate reshape nodes for the same input.
- Gradients: the pass-local golden fixture includes output, scalar loss, input
  gradient, and all eight MHA parameter gradients from the pre-lever code. The
  post-lever test matched it exactly.
- RNG: no random draw path changed during forward; module initialization remains
  untouched.

Golden fixture:

```text
path: artifacts/optimization/golden_outputs/ft_nn_mha_self_flat_reuse_frankentorch-l3mm.txt
sha256: daffba8149e995d8fb07284e5edcbd18f5593fbf8fd3c6852604faa1f0cabb15
```

## Result

After:

```text
worker: vmi1227854
command: rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_8x64x128_h8 --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [15.193 ms 15.566 ms 16.158 ms]
```

Delta by p50: `17.976 ms -> 15.566 ms`, about `13.4%` faster.

Confidence is capped because RCH selected a different worker for the after-run,
but the effect size is well above the overlapping-noise passes that were
rejected earlier in the campaign.

Score: Impact 2 * Confidence 2 / Effort 1 = 4.0.

Decision: keep.

## Gates

- PASS: `rch exec -- cargo test -p ft-nn mha_self_flat_reuse_golden_output_matches_fixture -- --nocapture`
- PASS: `rch exec -- cargo test -p ft-nn mha -- --nocapture` (8 MHA tests)
- PASS: `rch exec -- cargo bench -p ft-nn --bench nn_bench -- multihead_attention/forward_8x64x128_h8 --warm-up-time 1 --measurement-time 5 --sample-size 20` before and after
- PASS: `rch exec -- cargo fmt -p ft-nn --check`
- PASS: `rch exec -- cargo check -p ft-nn --all-targets`
- PASS: `rch exec -- cargo clippy -p ft-nn --all-targets --no-deps -- -D warnings`
- PASS: `git diff --check -- crates/ft-nn/src/lib.rs artifacts/optimization/golden_outputs/ft_nn_mha_self_flat_reuse_frankentorch-l3mm.txt`
- UBS: `ubs crates/ft-nn/src/lib.rs artifacts/optimization/golden_outputs/ft_nn_mha_self_flat_reuse_frankentorch-l3mm.txt` exited nonzero on existing `ft-nn/src/lib.rs` inventory: 100 equality-comparison false-positive criticals plus broad pre-existing warnings. UBS reported its built-in formatting, clippy, cargo check, test-build, cargo-audit, and cargo-deny probes clean.
