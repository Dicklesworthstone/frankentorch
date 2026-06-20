# frankentorch-o888p BOLD-VERIFY Scorecard

Agent: IvoryDeer / cod-b
Base: 5681ca0f
Date: 2026-06-20
Worktree: `/data/projects/.scratch/frankentorch-cod-b-bold-verify-20260620T041328Z`

## Lever

One kernel scheduling lever for the f64 scalar-loss path `sum(max_pool3d(input))`:

- Add a row-major unrolled `2x2x2/stride-2` leaf for `max_pool3d_sum_forward_with_indices_f64`.
- Stream scalar-loss sidecar offsets directly in `max_pool3d_backward_from_indices_scalar_f64` instead of recomputing `(oz, oy, ox)` coordinates.
- Leave the generic pairwise tree, first-tie argmax order, public API, and fallback paths intact.

Alien/perf mapping: this is the low-risk "cache/branch/loop-shape" lever from the graveyard guidance, not a new algorithm. It removes hot-path loop-control overhead and preserves the same row-major output order and pairwise split tree.

## Profile Before Change

Local PyTorch-enabled baseline log: `baseline_local_max_pool3d_20260620T041328Z.log`

| Row | Median | Signal |
| --- | ---: | --- |
| FrankenTorch materialized `max_pool3d -> sum -> backward` | 8.2494 ms | current public trace |
| FrankenTorch explicit fused scalar-loss path | 7.7994 ms | kernel lever target |
| PyTorch 2.12 CPU | 1.9145 ms | incumbent |
| Stage: `frankentorch_backward_only` | 6.4558 ms | largest traced stage |
| Stage: `kernel_forward_with_indices` | 817.61 us | lower-level kernel cost |
| Stage: `kernel_backward_from_indices` | 1.7696 ms | sidecar scatter cost |

Baseline fused ratio vs PyTorch: 7.7994 / 1.9145 = 4.074x slower.

## Final Head-to-Head

Local PyTorch-enabled after log: `after_local_max_pool3d_specialized_leaf_20260620T041328Z.log`

| Row | Before median | After median | Result |
| --- | ---: | ---: | --- |
| FrankenTorch materialized `max_pool3d -> sum -> backward` | 8.2494 ms | 7.9640 ms | neutral/no significant change |
| FrankenTorch explicit fused scalar-loss path | 7.7994 ms | 6.6646 ms | 1.170x faster, 14.5% lower median |
| PyTorch 2.12 CPU | 1.9145 ms | 1.8889 ms | incumbent stable/no significant change |

Final fused ratio vs PyTorch: 6.6646 / 1.8889 = 3.528x slower.

Strict ratio-vs-PyTorch verdict: loss. The lever improves FrankenTorch's local PyTorch ratio from 4.074x slower to 3.528x slower, but PyTorch still wins.

## Stage Movement

| Stage | Before median | After median | Result |
| --- | ---: | ---: | --- |
| `frankentorch_backward_only` | 6.4558 ms | 5.2178 ms | 1.237x faster |
| `kernel_forward_with_indices` | 817.61 us | 721.37 us | 1.133x faster |
| `kernel_backward_from_indices` | 1.7696 ms | 1.5417 ms | 1.148x faster |

The setup-tensor row regressed in this noisy run (209.35 us -> 237.30 us), but it is outside the changed kernel path and the end-to-end fused row still improved.

## Remote rch Evidence

Remote Rust-only after bench on `hz2`: `after_rch_fused_sum_specialized_leaf_20260620T041328Z.log`

| Row | Median | Context |
| --- | ---: | --- |
| `gauntlet_max_pool3d_grad/frankentorch_fused_sum_loss` | 6.3252 ms | after row on `hz2` |

Prior committed `frankentorch-nzqb9` scorecard recorded `hz2` fused median at 8.5813 ms, so this is consistent with a same-worker improvement. The local PyTorch-enabled run remains the primary proof because remote workers do not have the local PyTorch venv.

## Correctness / Gates

| Gate | Status | Artifact |
| --- | --- | --- |
| Kernel bit-equivalence test | green | `test_kernel_max_pool3d_scalar_specialized_20260620T041328Z.log` |
| API bit-equivalence/autograd test | green | `test_api_max_pool3d_sum_bits_20260620T041328Z.log` |
| `cargo check -p ft-kernel-cpu --all-targets` | green with pre-existing example warnings | `check_ft_kernel_cpu_20260620T041328Z.log` |
| `cargo test -p ft-conformance` | green | `test_ft_conformance_20260620T041328Z.log` |
| `git diff --check` | green | command output empty |
| `cargo fmt --check` | red, broad pre-existing workspace formatting diffs | `fmt_check_20260620T041328Z.log`, `fmt_check_status_20260620T041328Z.log` |
| `cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings` | red, pre-existing lint debt outside this change | `clippy_ft_kernel_cpu_20260620T041328Z.log` |
| `ubs crates/ft-kernel-cpu/src/lib.rs` | zero criticals; warning inventory remains | `ubs_ft_kernel_cpu_20260620T041328Z.log` |

## Isomorphism Proof

- Ordering preserved: yes. The specialized leaf visits each output in the same row-major order as the generic leaf and writes the same first-tie argmax offset.
- Tie-breaking unchanged: yes. Candidate comparisons remain strict `>`, so equal maxima keep the first location.
- Floating-point drift: identical for the covered bit-equivalence tests. The fused sum still uses the existing recursive pairwise split tree; only leaf max selection is unrolled for `2x2x2/stride-2`.
- RNG seeds: N/A.
- Golden outputs: focused kernel/API bit-equivalence tests and `ft-conformance` are green.
- Rollback plan: revert the commit.

## Verdict

Keep the kernel trim. It is a measured local fused-path win, green on conformance, and improves the PyTorch ratio, but it does not beat PyTorch.

Follow-up for the remaining loss: `frankentorch-kfdnn` targets automatic graph fusion, session/autograd allocation churn, sidecar representation, or deeper scheduling.
