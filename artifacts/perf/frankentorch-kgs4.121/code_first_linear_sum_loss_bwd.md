# frankentorch-kgs4.121 - Linear sum-loss backward all-ones dy fast path

Status: code-first, batch-test pending.

Agent: cod-a / IvoryDeer

## Target

Profile source:

- `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`
- `linear_train/hidden/2048`: `[37.335 ms 39.913 ms 42.149 ms]`

Benchmark contract:

- `bench_linear_train` builds f64 `[32, 512] @ [hidden, 512]^T + bias`.
- It reduces the output with `tensor_sum(y)`.
- Therefore first-order backward sees `dy == 1.0` for every output element.

## Lever

The normal first-order f64 Linear backward does:

1. `dx = dy @ weight`
2. materialize `dy^T`
3. `dweight = dy^T @ x`
4. `dbias = sum_batch(dy)`

For exact all-ones `dy`, those reduce to:

1. one `weight` column-sum row repeated across `batch`
2. one `x` column-sum row repeated across `out_features`
3. constant `dbias[o] = batch`

This pass adds the exact all-ones detector in the f64 `functional_linear`
first-order backward closure and routes that training trace to the closed-form
helper. Non-unit `dy` falls back to the existing `ft_kernel_cpu::linear_backward_f64`.

The ideal kernel-level placement is temporarily avoided because
`crates/ft-kernel-cpu/src/lib.rs` is actively leased by `OrangeCedar` for
`frankentorch-kgs4.120`.

## Guard

Added:

- `linear_backward_all_ones_dy_matches_kernel_reference`

The guard compares the closed-form helper with the existing generic
`linear_backward_f64` reference on a non-square shape, including `dx`,
`dweight`, and `dbias`.

Local verification allowed by the campaign:

```text
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a cargo check -p ft-api
```

Criterion, conformance, tests, clippy, fmt, and rch are intentionally pending
for the batch gate.

## Negative-Evidence Ledger

| Attempt | Evidence | Decision | Do not retry |
| --- | --- | --- | --- |
| Borrowed Linear grad inputs | `artifacts/perf/frankentorch-t1vg/report.md` | Same-worker `linear_train/hidden/2048` improved only `1.061x`, Score `1.01`; rejected below keep gate | First-order save/borrow micro-tuning as the main Linear lever |
| Persistent f64 Linear transpose cache | `artifacts/perf/frankentorch-kgs4.56/rejected_persistent_linear_weight_cache.md` | Regressed three of four forward rows; rejected | Normal-GEMM transpose caching for Linear |
| Recursive Strassen dgemm pilot | `.beads/issues.jsonl` entry `frankentorch-1e7c` | Slower p50; source reverted | Naive Strassen without preallocated/fused workspace |
| RMSNorm unit-dy stat staging | `artifacts/perf/frankentorch-t89dc/closeout_rms_norm_unit_dy_reject.md` | Ambiguous `p = 0.58`; rejected | Norm stat-staging micro-levers |
| Pooling direct/duplicate lanes | `artifacts/perf/frankentorch-pool2d-borrowed-max/report.md`, `artifacts/perf/frankentorch-3oyr5/closeout_rejected_duplicate_avg_pool2d_2x2s2.md` | Rejected/duplicate | 2D pool borrowed/direct retries without a new profile row |

## Radical-Lever Rationale

The alien-graveyard filter points at cache-aware layout, vectorized execution,
and communication-avoiding linear algebra. This is the smallest safe version of
that idea: avoid communicating repeated all-ones rows through GEMM and avoid
materializing `dy^T` at all for the benchmark's sum-loss path.

Expected speed mechanism:

- remove two GEMM calls from the sum-loss Linear backward;
- remove the `dy^T` allocation and write;
- replace them with two linear reductions plus parallel row fills.

No speedup is claimed until a focused same-worker Criterion run and conformance
batch confirm it.
