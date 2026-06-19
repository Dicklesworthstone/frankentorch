# frankentorch-kgs4.113 SDPA scaled GEMM alpha scorecard

Date: 2026-06-19
Agent: IvoryDeer / cod-b
Worker for Rust A/B: `vmi1227854`

## Lever

Keep the already-present SDPA backward scaled GEMM-alpha path:

- `dQ = scale * dU @ K` via `dgemm_scaled` / `sgemm_scaled`.
- `dK = scale * dU^T @ Q` via `dgemm_tb_scaled` / `sgemm_tb_scaled`.
- Reject the old post-GEMM `for v { *v *= scale }` full-buffer streams.

The alien-lever mapping is cache-local numeric kernels plus vectorized/morsel
execution discipline: remove two avoidable memory streams from the training
hot path instead of adding another scalar pass around a BLAS-shaped primitive.

## Measurements

| Row | Host | Command | Median | Verdict |
| --- | --- | --- | ---: | --- |
| Current scaled alpha | rch `vmi1227854` | `cargo bench -p ft-api --bench ops_bench -- sdpa/grad_16x512x64 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | `82.730 ms` | Keep baseline |
| Temporary old post-scale | rch `vmi1227854` | same command after measurement-only post-scale patch | `114.40 ms` | Rejected |
| Remote FT gauntlet arm | rch `vmi1227854` | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- sdpa --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | `53.254 ms` | Rust-only evidence |
| Local FT gauntlet arm | local | same gauntlet with `/tmp/torchvenv/bin/python` | `63.057 ms` | Ratio input |
| Local PyTorch gauntlet arm | local | same gauntlet, PyTorch `2.12.0+cpu` | `48.915 ms` | Ratio reference |

Same-worker Rust A/B: `82.730 / 114.40 = 0.723x`; scaled alpha is `1.38x`
faster than old post-scale.

PyTorch ratio: `63.057 / 48.915 = 1.29x` slower. Win/loss/neutral vs PyTorch:
`0W / 1L / 0N`.

## Caveats

Remote PyTorch was not available through rch on `vmi1227854`. The pinned remote
gauntlet built and ran the FrankenTorch arm, then the PyTorch subprocess failed
with `ModuleNotFoundError: No module named 'torch'`. The local PyTorch ratio is
diagnostic; the keep/reject decision uses the same-worker rch Rust A/B.

## Gates

- `rch exec -- cargo test -p ft-kernel-cpu scaled_gemm_matches_post_scale_reference -- --nocapture`:
  passed on `hz2` (`1 passed; 0 failed`).
- `rch exec -- cargo test -p ft-api sdpa_ -- --nocapture`: passed on `hz2`
  (`17` SDPA unit tests plus `sdpa_backward_grads_match_finite_diff`).
- `rch exec -- cargo test -p ft-conformance`: passed on `vmi1149989`
  (`199` lib tests plus bins, E2E, PyTorch conformance, smoke, and doc-tests).
- `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench`:
  passed on `hz1`.
- `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings`:
  passed on `hz1`.
- `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs`:
  passed; `/tmp/torchvenv/bin/python -m py_compile
  crates/ft-api/benches/pytorch_sdpa_grad.py` passed; `git diff --check`
  passed.
- `ubs` on changed source/docs/artifact summaries: zero critical or warning
  findings.

Known caveat: broad `cargo fmt --check` still reports pre-existing formatting
drift across unrelated ft-api examples and large source files. This closeout
does not apply workspace-wide format churn.
