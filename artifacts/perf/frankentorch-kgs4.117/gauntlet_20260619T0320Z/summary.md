# frankentorch-kgs4.117 max_pool3d gauntlet result

Agent: IvoryDeer / cod-b
Date: 2026-06-19

## Workload

`gauntlet_max_pool3d_grad`: deterministic f64 `[2,32,16,32,32]`, kernel
`2x2x2`, stride `2x2x2`, `functional_max_pool3d`, scalar `sum`, backward.
PyTorch reference: `/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`,
torch `2.12.1+cpu`, 32 compute threads and 32 interop threads.

## Results

| Row | FrankenTorch median | PyTorch median | FT/PyTorch | Verdict |
|---|---:|---:|---:|---|
| Parent `c79d3a23` | `20.585 ms` | `2.1381 ms` | `9.63x` slower | before sidecar |
| Current post-lint | `15.794 ms` | `1.6228 ms` | `9.73x` slower | internal keep, PyTorch loss |

Internal speedup: `20.585 / 15.794 = 1.30x`.

Supplemental rch row: `hz2` built the bench and measured current FrankenTorch at
`28.124 ms`, but the PyTorch arm failed there with `ModuleNotFoundError: No
module named 'torch'`; this is not used for ratio scoring.

## Decision

Keep the max_pool3d saved-index sidecar as a measured internal win. Do not count
it as release-ready PyTorch dominance. No source revert.

Retry condition: do not retry max_pool3d sidecar-only or rescan-only variants
unless a fresh profile proves saved-context memory or backward window rescans
still dominate after session setup, allocation churn, and tensor materializing
costs are separated.

## Evidence

- `current_local_warm_postlint_criterion.txt`
- `parent_local_warm_criterion.txt`
- `current_criterion.txt`
- `ft_api_bench_check.log`
- `ft_kernel_cpu_max_pool3d_sidecar_test_postlint.log`
- `ft_api_max_pool3d_grad_test_postlint.log`
- `ft_api_bench_clippy_postlint.log`
