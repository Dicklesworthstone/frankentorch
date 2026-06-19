# frankentorch-kgs4.128 max_pool3d BOLD-VERIFY summary

Agent: IvoryDeer (`cod-b`)
Date: 2026-06-19
Host: `thinkstation1`
PyTorch: `2.12.1+cpu`, 32 compute threads, 32 interop threads

## Clean baseline

Command:

```bash
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d --noplot
```

Result:

- FrankenTorch median: `15.303 ms`
- PyTorch median: `1.6325 ms`
- Ratio: FrankenTorch `9.38x` slower

## Stage probe

Temporary diagnostic rows added to `pytorch_gauntlet_bench.rs`:

- setup tensor: `215.47 us`
- forward-only: `4.1256 ms`
- sum-only: `1.3121 ms`
- backward-only: `43.433 ms` with severe outliers
- raw kernel forward+indices: `727.15 us`
- raw kernel backward-from-indices: `9.0069 ms` with severe outliers

Use this as routing evidence only. It showed a large gap between the API
forward path and raw forward kernel, and noisy but substantial backward cost.

## Rejected candidates

1. Borrowed-input f64 custom-op route for max_pool3d.
   - Isolated forward improved to `1.8935 ms`.
   - Full headline regressed to `22.764 ms` vs PyTorch `1.6633 ms`.
   - Ratio: `13.69x` slower; `1.49x` slower than clean baseline.
   - Reverted.

2. Rayon unit-`dout` backward scatter from saved argmax offsets.
   - Full headline: FrankenTorch `16.160 ms`, PyTorch `1.6543 ms`.
   - Ratio: `9.77x` slower; `1.06x` slower than clean baseline.
   - Reverted.

3. Sequential unit-`dout` backward scatter.
   - Full headline: FrankenTorch `22.465 ms`.
   - Paired PyTorch row had severe high outliers, so use clean PyTorch
     baseline for routing: `13.76x` slower.
   - Reverted.

Final restored-source sanity row measured FrankenTorch `16.586 ms`; its paired
PyTorch row had severe high outliers and is not primary ratio evidence.

## Verdict

No product source was kept. The next max_pool3d attempt should not retry
borrowed-input-only routing or standalone unit-`dout` scatter. The profile points
toward end-to-end fusion that removes the sum-generated gradient buffer/tape
edge, allocator/arena changes measured on the whole row, or a different
layout/kernel plan.
