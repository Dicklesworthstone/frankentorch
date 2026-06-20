# frankentorch-nzqb9 BOLD-VERIFY Scorecard

Agent: IvoryDeer / cod-b
Base: 45c2e011
Date: 2026-06-20
Worktree: /data/projects/.scratch/frankentorch-cod-b-bold-verify-20260620T020738Z

## Lever

Fused f64 5D `sum(max_pool3d(input))` scalar-loss path:

- `ft-kernel-cpu::max_pool3d_sum_forward_with_indices_f64`
- `ft-kernel-cpu::max_pool3d_backward_from_indices_scalar_f64`
- `FrankenTorchSession::functional_max_pool3d_sum`
- Criterion row `gauntlet_max_pool3d_grad/frankentorch_fused_sum_loss`

The final implementation keeps the pairwise sum split tree, stores the same first-tie argmax sidecar, avoids the dense pooled output node and dense output-gradient buffer, increments row-major coordinates inside leaf blocks, and uses a lower fused-pool parallel threshold because each element performs window work.

## Final Head-to-Head

Local PyTorch-enabled run:

Log: `artifacts/perf/frankentorch-nzqb9/local_pytorch_ratio_max_pool3d_fused_sum_loss_v2_20260620T020738Z.log`

| Row | Median | Result |
| --- | ---: | --- |
| FrankenTorch materialized `max_pool3d -> sum -> backward` | 6.5281 ms | baseline |
| FrankenTorch fused scalar-loss path | 5.0290 ms | 1.298x faster, 23.0% lower median |
| PyTorch 2.12 CPU | 1.9905 ms | still 2.527x faster than fused FrankenTorch |

Strict ratio-vs-PyTorch verdict: loss. The lever improves FrankenTorch's local PyTorch ratio from 3.280x slower to 2.527x slower, but PyTorch still wins.

## Same-Worker Rust Proof

Final `rch` run on `hz2`:

Log: `artifacts/perf/frankentorch-nzqb9/after_hz2_max_pool3d_fused_sum_loss_v2_20260620T020738Z.log`

| Row | Median | Result |
| --- | ---: | --- |
| `frankentorch_kgs4_117` | 10.363 ms | baseline |
| `frankentorch_fused_sum_loss` | 8.5813 ms | 1.208x faster, 17.2% lower median |

Earlier `hz2` routing proof before the final leaf/threshold revision was also positive:

- `after_hz2_max_pool3d_fused_sum_loss_20260620T020738Z.log`: 7.9285 ms -> 5.5053 ms, 1.440x faster.
- `confirm_hz2_max_pool3d_fused_sum_loss_20260620T020738Z.log`: 7.9815 ms -> 6.0056 ms, 1.329x faster.

## Gates

| Gate | Status | Artifact |
| --- | --- | --- |
| Kernel bit-equivalence test | green | `test_kernel_max_pool3d_sum_scalar_v2_20260620T020738Z.log` |
| API bit-equivalence/autograd test | green | `test_api_max_pool3d_sum_fused_v2_20260620T020738Z.log` |
| `cargo check -p ft-api --bench pytorch_gauntlet_bench` | green | `check_ft_api_bench_v2_20260620T020738Z.log` |
| `cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings` | green after two surgical pre-existing range-check fixes | `clippy_ft_api_bench_final_20260620T020738Z.log` |
| `cargo test -p ft-conformance` | green | `test_ft_conformance_v2_20260620T020738Z.log` |
| `git diff --check` | green | command output empty |
| `cargo fmt --check` | red, broad pre-existing workspace formatting diffs | `fmt_check_v2_20260620T020738Z.log` |
| `ubs <changed Rust files>` | red, broad pre-existing inventory across huge files; UBS's embedded clippy/check probes were green | `ubs_changed_files_final_20260620T020738Z.log` |

`cargo fmt --check` was not auto-applied because it reports broad unrelated churn across existing files and large unrelated regions. New overlapping hunks in the fused kernel/test were manually formatted.

## Verdict

Keep the final additive fused primitive and bench row.

The commit does not auto-rewrite existing `functional_max_pool3d(...); tensor_sum(...)` calls, so existing behavior is unchanged. The new path is opt-in, bit-equivalent to materialized pool+sum for the tested f64 path, green on conformance, and improves the local PyTorch ratio even though it does not beat PyTorch.

Next route: attack the remaining 2.527x PyTorch gap through session/autograd allocation churn and automatic graph fusion for `max_pool3d -> sum -> backward`, with same-machine PyTorch ratio proof as the keep gate.
