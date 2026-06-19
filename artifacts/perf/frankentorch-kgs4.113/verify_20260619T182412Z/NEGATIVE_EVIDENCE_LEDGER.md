# frankentorch-kgs4.113 negative-evidence ledger

## Rejected

| Attempt | Evidence | Verdict |
| --- | --- | --- |
| SDPA backward post-GEMM full-buffer scale streams | Same-worker rch `vmi1227854`: old post-scale median `114.40 ms` vs current scaled-alpha median `82.730 ms`; Criterion marked post-scale as `[+21.885% +37.179% +55.712%]`, `p=0.00`. | Reject; keep scaled alpha. |
| Treat remote PyTorch failure as a valid ratio | Pinned rch gauntlet ran FrankenTorch at `53.254 ms` but failed the PyTorch arm with `ModuleNotFoundError: No module named 'torch'`. | Reject as PyTorch ratio evidence; use only as remote build/FT-arm evidence. |

## PyTorch Gap

Local diagnostic gauntlet with PyTorch `2.12.0+cpu`:

- FrankenTorch median: `63.057 ms`
- PyTorch median: `48.915 ms`
- Ratio vs PyTorch: `1.29x` slower
- Win/loss/neutral: `0W / 1L / 0N`

## Do Not Retry

Do not return to a separate post-GEMM scale pass for SDPA `dQ`/`dK`. Future
work should attack the remaining gap with a deeper whole-row plan: softmax/GEMM
cache blocking, f32-native ratio work, tape/allocation arena cuts, or fused
loss/backward scheduling.
