# FrankenTorch Release-Readiness Scorecard

Updated: 2026-06-19

## Performance Gauntlet

| Bead | Workload | Result vs PyTorch | Before/after verdict | Release action |
|---|---:|---:|---:|---|
| `frankentorch-kgs4.117` | max_pool3d f64 train step `[2,32,16,32,32]` | `9.73x` slower | internal keep; `20.585 ms` -> `15.794 ms`; remote PyTorch arm unavailable on `hz2` | kept; profile deeper end-to-end gap |
| `frankentorch-kgs4.121` | linear f64 train step `[32,512] -> 2048` | `2.45x` slower | API-local internal keep; `29.606 ms` -> `22.775 ms`; kernel move `26.459 ms` rejected | kept API helper; reverted kernel move |
| `frankentorch-kgs4.122` | avg_pool1d f64 train step `[8,64,8192]` | `25.86x` slower; final rerun `24.92x` slower | no gain; candidate median `204.02 ms` vs fast-path-disabled `179.91 ms` | reverted |
| `frankentorch-kgs4.124` | SmoothL1 f64 mean-loss backward, 8M elems | `1.99x` slower | internal keep; `963.16 ms` -> `757.63 ms` on `hz2` | kept; follow-up `frankentorch-kgs4.128` |
| `frankentorch-kgs4.126` | max_pool1d f64 train step `[8,64,8192]` | `12.31x` slower | no gain; candidate median `184.41 ms` vs parent `178.47 ms` | reverted |

Score: `5/5` for the measured gauntlet lanes. Correctness guards are green and
the MaxPool3d, Linear, and SmoothL1 levers are real internal speedups, but no
measured workload is performance-dominant against PyTorch yet.

## Current Gates

| Gate | Scope | Result |
|---|---|---|
| Criterion | `cargo bench -p ft-api --bench ops_bench -- smooth_l1/grad_8m --noplot` | completed on `hz2`; current median `757.63 ms` |
| PyTorch oracle | `torch_smooth_l1_grad_8m.py` | local PyTorch `2.12.1+cu130` median `373.61 ms` |
| Compile | `rch exec -- cargo check -p ft-api` | passed |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu smooth_l1_backward_reduced_f64_matches_uniform_dloss_bits` | passed |
| Correctness | `rch exec -- cargo test -p ft-api smooth_l1_loss_reduced_grad_matches_materialized_reference_bits` | passed after explicit default-argument test-call fix |
| Criterion | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool1d --noplot` | completed locally with PyTorch `2.12.1+cpu` |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `ovh-a` for final harness |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu max_pool1d_direct_matches_2d_h1_first_tie_forward_backward_bit_exact` | passed on `ovh-a` |
| Formatting | `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs` | passed |
| Criterion | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- avg_pool1d --noplot` | completed locally; candidate FrankenTorch median `204.02 ms`, PyTorch median `7.4798 ms`; final reverted FrankenTorch median `184.99 ms`, PyTorch median `7.1539 ms`; rerun final median `181.94 ms`, PyTorch median `7.3011 ms` |
| Remote build/bench | `rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- avg_pool1d --noplot` | Rust bench built and ran on `vmi1227854`; PyTorch arm failed because remote `torch` was unavailable, so this is not ratio evidence |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `vmi1152480` after avg_pool1d revert |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu avg_pool1d_direct_matches_2d_h1_forward_backward_bit_exact -- --nocapture` | passed on `ovh-a` after avg_pool1d revert |
| Correctness | `rch exec -- cargo test -p ft-api functional_avg_pool1d_fused_matches_reshape_2d_forward_and_backward_bits -- --nocapture` | passed on `vmi1227854` after avg_pool1d revert |
| Clippy | `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings` | passed on `hz1` after avg_pool1d revert |
| Clippy | `rch exec -- cargo clippy -p ft-kernel-cpu -- -D warnings` | passed on `vmi1264463` after avg_pool1d revert |
| UBS | `ubs crates/ft-api/benches/pytorch_gauntlet_bench.rs crates/ft-kernel-cpu/src/lib.rs crates/ft-api/benches/pytorch_avg_pool1d_grad.py` | no critical findings; existing `ft-kernel-cpu` warning inventories remain outside this revert |
| Formatting | `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs`; `git diff --check` on changed files | passed; whole-file `ft-kernel-cpu/src/lib.rs` rustfmt check still reports pre-existing formatting drift outside this avg_pool1d revert |
| Criterion | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- linear --noplot` | completed locally; final FrankenTorch median `22.775 ms`, PyTorch median `9.2821 ms` |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `vmi1152480` after restoring the API-local linear path |
| Correctness | `rch exec -- cargo test -p ft-api linear_backward_all_ones_dy_matches_kernel_reference -- --nocapture` | passed on `vmi1293453` |
| Criterion | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d --noplot` | completed locally; current FrankenTorch median `15.794 ms`, PyTorch median `1.6228 ms` |
| Criterion baseline | parent worktree at `c79d3a23`, same max_pool3d harness | completed locally; parent FrankenTorch median `20.585 ms`, PyTorch median `2.1381 ms` |
| Remote build/bench | `rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d --noplot` | built on `hz2`; current FrankenTorch median `28.124 ms`; PyTorch arm failed because remote `torch` was unavailable |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `hz2` |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu max_pool3d_indices_scatter_matches_rescan_first_tie_bits` | passed after clippy-only lint fixes |
| Correctness | `rch exec -- cargo test -p ft-api functional_max_pool3d_grad_matches_finite_diff` | passed after clippy-only lint fixes |
| Clippy | `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings` | passed after narrow ft-kernel-cpu clippy fixes |

Known caveat: `cargo fmt --check -p ft-api` remains blocked by pre-existing
crate-wide formatting debt in unrelated examples and long `ft-api/src/lib.rs`
regions. `cargo fmt -p ft-api -- --check` also reports broad existing drift
for the SmoothL1 closeout; follow-up `frankentorch-6xsy8` tracks that cleanup.

UBS caveat: a full changed-file UBS scan including the 136k-line
`ft-api/src/lib.rs` did not complete after several minutes and was interrupted
or timed out in the pre-commit hook. The max_pool1d benchmark surface was
scanned directly and passed with zero critical or warning findings; no UBS
verdict was available for the SmoothL1 `ft-api/src/lib.rs` closeout.

## Next Perf Target

The `.124` result points toward deeper SmoothL1 training overhead: tape setup,
input/materialization cost, loss backward kernel shape, SIMD, and cache layout.
The `.122` result points away from tiny avg_pool1d unit-gradient fill branches
and toward end-to-end pooling overhead, especially session/tape setup,
allocation churn, forward materialization, and generic pooling dispatch costs.
The `.126` result points away from tiny max_pool1d backward scatter branches and
toward larger full-step costs: autograd/session setup, allocation churn, and
forward saved-index materialization. The `.117` result keeps compact max_pool3d
sidecars as an internal win but leaves a roughly 10x PyTorch gap, so the next
pooling pass needs an end-to-end profile rather than another sidecar-only tweak.
The `.121` result points away from moving the all-ones linear backward shortcut
into the generic CPU kernel; the remaining gap is end-to-end linear training
overhead, not the already-collapsed row-sum/copy helper alone. Future work
should profile those frames before trying another scalar-wrapper or one-off
unit-gradient branch.
