# frankentorch-kgs4 SDPA scaled GEMM alpha code-first attempt

Date: 2026-06-18
Agent: IvoryDeer / cod-a
Status: code-first, batch-test pending

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
| SDPA dQ/dK GEMM alpha scaling | This attempt | Local cargo check only by instruction; Criterion/conformance batch pending. | Pending measurement; revert if SDPA grad or conformance regresses. |

## Required batch follow-up

When batch tests are allowed, run the focused SDPA grad criterion before and
after on the same worker and preserve:

- `sdpa/grad_16x512x64` f64/f32 deltas if available.
- `cargo test -p ft-kernel-cpu scaled_gemm_matches_post_scale_reference`.
- Any SDPA conformance/golden guard that covers backward outputs.

No performance win is claimed until those measurements land.
