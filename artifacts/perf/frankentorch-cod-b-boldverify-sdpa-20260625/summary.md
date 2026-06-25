# frankentorch cod-b BOLD-VERIFY SDPA f64 unit-dout rejection

Agent: PearlReef
Date: 2026-06-25
Head: `e2642b5c` (`origin/main`)
Target dir requested: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b`

## Worktree Gate

Fresh scan found no clean unlanded measured win to land. The only worktree ahead
of `origin/main` was `/data/projects/frankentorch-gxpb2-pass10`, whose head is an
explicit large-n row-SIMD rejection.

## Lever

Tried a f64 mirror of the existing f32 `sdpa_backward_f32_unit_dout` scalar-loss
shortcut. For exact all-ones upstream gradients (`sum(SDPA).backward()`), it
replaced `dout @ V^T` and `P^T @ dout` with row/column reductions while leaving
the `dQ` and `dK` GEMMs unchanged.

Mapped graveyard/artifact sources:

- `alien_cs_graveyard.md`: profile-first rule, constants-kill-you failure mode,
  vectorized/cache-local kernel discipline.
- `high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md`:
  SIMD/tiled kernels plus proof artifacts and profile-first optimization gates.
- `alien-artifact-coding`: algebraic artifact with behavior-preservation proof
  and deterministic fallback to the dense backward for non-all-ones gradients.

EV before measurement: Impact 3, Confidence 4, Reuse 4, Effort 2, Friction 2,
EV = 12.0. Rejected after measurement because the PyTorch win bar was not met.

## Bench Evidence

The user-requested exact command form was tried first:

```text
AGENT_NAME=PearlReef PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo bench --release -p ft-api --bench pytorch_gauntlet_bench -- sdpa --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

This Cargo rejects `cargo bench --release` (`unexpected argument '--release'`);
see `baseline_sdpa_exact_cargo_bench_release.log`.

Accepted per-crate bench command:

```text
AGENT_NAME=PearlReef CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- sdpa --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Remote PyTorch rows failed on both workers because worker Python had no `torch`;
FT Criterion rows still completed:

- Baseline FT (`vmi1227854`, detached `origin/main` worktree): `[25.351 ms 26.339 ms 28.071 ms]`.
- Candidate FT (`hz2`, main checkout with source candidate): `[21.214 ms 22.289 ms 23.671 ms]`.
- Internal FT delta by midpoint: `26.339 / 22.289 = 1.18x` faster.

Local PyTorch sidecar for the same script and fixture:

- Captured run: `0.397009555018 s / 20 = 19.850 ms`, checksum `0.103877428238`.
- First uncaptured same-session run: `0.362935346086 s / 20 = 18.147 ms`, checksum `0.103877428238`.

FT/PyTorch ratio:

- Baseline: `26.339 / 18.147 = 1.45x slower` by best local PyTorch, or
  `26.339 / 19.850 = 1.33x slower` by captured local PyTorch.
- Candidate: `22.289 / 18.147 = 1.23x slower` by best local PyTorch, or
  `22.289 / 19.850 = 1.12x slower` by captured local PyTorch.

## Decision

REJECT and revert source. The algebraic shortcut is a real internal improvement,
but not a PyTorch win, and the user gate requires a PyTorch-facing win. Do not
retry f64 SDPA unit-dout reduction as a standalone lever unless the lower-level
same-worker PyTorch comparator is available and the candidate clears `<1.0x`
FT/PyTorch, or a deeper fused forward+backward pass removes more than the two
all-ones GEMMs.

## Behavior Preservation

- Ordering preserved: yes. Each softmax row still computes max-subtract,
  exponentiation, normalization, and `dU` in the same row order.
- Tie-breaking: N/A for this dense differentiable kernel.
- Floating point: `dQ` and `dK` GEMMs unchanged; `dP`/`dV` specialize exact
  all-ones `dout` algebraically.
- Fallback: non-all-ones upstream gradients still use `sdpa_backward_f64`.
- Source disposition: reverted; no product code retained.
