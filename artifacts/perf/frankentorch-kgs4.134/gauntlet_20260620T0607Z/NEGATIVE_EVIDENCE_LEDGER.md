# frankentorch-kgs4.134 Negative-Evidence Entry

- Lever: fused f64 `sum(avg_pool1d(input, kernel=2, stride=2))` scalar-loss
  forward/backward.
- Workload: `pytorch_gauntlet_bench` `avg_pool1d`, f64
  `[N,C,L]=[8,64,8192]`, scalar `sum` loss.
- Baseline local PyTorch oracle: old FT `79.285 ms`, PyTorch `6.2886 ms`,
  ratio `12.61x` slower.
- Candidate local PyTorch oracle: old FT `69.267 ms`, fused FT `59.050 ms`,
  PyTorch `7.8192 ms`; fused/old `0.8525x`, fused/PyTorch `7.55x` slower.
- Candidate rch Rust-only run on `vmi1152480`: old FT `134.74 ms`, fused FT
  `87.564 ms`; fused/old `0.6500x`. Remote PyTorch failed because `torch` was
  unavailable on the worker.
- Win/loss/neutral vs PyTorch: `0W / 1L / 0N`.
- Verdict: keep as a measured internal win, still a PyTorch-loss row.
- Retry condition: route future avg_pool1d work to allocation/tape/loss-fusion
  primitives or whole-buffer `.grad` traffic removal, not another pool-kernel
  microlever.
