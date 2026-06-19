# frankentorch-kgs4.113 SDPA scaled GEMM alpha code-first attempt

Date: 2026-06-18
Agent: IvoryDeer / cod-b
Status: measured keep on 2026-06-19
Bead: frankentorch-kgs4.113

## Lever

Fold SDPA backward's final `scale` multiply for `dQ` and `dK` into the GEMM
alpha parameter:

- `dQ = scale * dU @ K` now calls `dgemm_scaled` / `sgemm_scaled`.
- `dK = scale * dU^T @ Q` now calls `dgemm_tb_scaled` / `sgemm_tb_scaled`.
- The post-GEMM full-buffer scale passes are removed.

This targets realistic transformer training shapes where SDPA backward already
uses the fused recompute path and spends two extra memory streams over
`[seq_q, d_k]` and `[seq_k, d_k]` per batch/head after the GEMM kernels finish.

## Correctness guard

Added `scaled_gemm_matches_post_scale_reference` in `ft-kernel-cpu`:

- f64 normal GEMM scaled alpha vs old GEMM-then-multiply reference.
- f64 transposed-left GEMM scaled alpha vs old GEMM-then-multiply reference.
- f32 mirrors for both paths.

The guard is intentionally small and serial-sized so it checks the semantic
contract without turning the batch suite into another benchmark.

## Negative-evidence ledger

| Attempt | Scope | Evidence | Status |
| --- | --- | --- | --- |
| SDPA backward transpose materialization removal | `frankentorch-kgs4.111` | Existing artifact `pass1_local_baseline_sdpa_grad.log` showed `sdpa/grad_16x512x64` improved from the prior materialized-transpose path. | Already applied; do not repeat. |
| Per-call packed f64 `dgemm_bt` panel | `artifacts/perf/frankentorch-kgs4-next/kgs4_53_packed_bt_panel_rejected.md` | Same-worker regressions / mixed results. | Rejected; do not retry per-call BT packing. |
| Per-call packed f32 `sgemm_bt` panel | `artifacts/perf/frankentorch-nfvtp/rejected_sgemm_bt_packed_panel.md` | Regressed f32 linear BT shapes. | Rejected; do not retry per-call BT packing. |
| Persistent linear weight cache | `artifacts/perf/frankentorch-kgs4.56/rejected_persistent_linear_weight_cache.md` | Existing rejection artifact. | Rejected; do not route SDPA through persistent weight cache. |
| SDPA dQ/dK GEMM alpha scaling | `frankentorch-kgs4.113` | Follow-up evidence in `artifacts/perf/frankentorch-kgs4.113/verify_20260619T182412Z/`: same-worker rch `vmi1227854` current scaled-alpha median `82.730 ms` vs temporary old post-scale median `114.40 ms` (`0.723x` latency, `1.38x` faster). Local PyTorch diagnostic ratio remains a loss: FrankenTorch `63.057 ms` vs PyTorch `48.915 ms` (`1.29x` slower). | Measured keep internally; PyTorch-loss row for release readiness. |

## Batch follow-up

Completed 2026-06-19 in
`artifacts/perf/frankentorch-kgs4.113/verify_20260619T182412Z/`.

Remote PyTorch caveat: rch built and ran the FrankenTorch gauntlet arm on
`vmi1227854`, but the PyTorch subprocess failed because that worker did not
have `torch` installed. The local PyTorch ratio is diagnostic; the same-worker
Rust A/B is the keep proof.
