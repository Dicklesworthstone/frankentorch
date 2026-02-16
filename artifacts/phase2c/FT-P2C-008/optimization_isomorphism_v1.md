# FT-P2C-008 Optimization + Isomorphism Evidence (v1)

## Optimization Lever

- ID: `nn-state-packet-e2e-fixture-cache-fastpath`
- Change: run packet-008 E2E through the fixture-cached nn_state emitter path and benchmark against the legacy emitter path to preserve deterministic replay while reducing packet latency tails.
- Path:
  - `crates/ft-conformance/src/lib.rs`

## Benchmark Delta (`packet_e2e_microbench_nn_state_legacy_vs_optimized_profiles`)

- Baseline (legacy emitter): `p50=1869569ns`, `p95=2028345ns`, `p99=2028345ns`, `mean=1850211ns`
- Post (optimized emitter): `p50=1590103ns`, `p95=1834217ns`, `p99=1834217ns`, `mean=1629710ns`
- Improvement: `p50=14.948% reduction`, `p95=9.571% reduction`, `p99=9.571% reduction`, `mean=11.918% reduction`

## Isomorphism Checks

- packet-specific e2e filter behavior: `e2e_matrix_packet_filter_includes_nn_state_packet_entries` passed
- packet-specific benchmark consistency: `packet_e2e_microbench_nn_state_produces_percentiles` and legacy-vs-optimized comparison passed
- nn_state conformance parity gates: `strict_nn_state_conformance_is_green` + `hardened_nn_state_conformance_is_green` passed
- differential packet evidence remains green via:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
  - `artifacts/phase2c/FT-P2C-008/differential_packet_report_v1.json`
  - `artifacts/phase2c/FT-P2C-008/differential_reconciliation_v1.md`

## Acceptance Note

This lever is accepted for FT-P2C-008 because it lowers packet-level strict/hardened E2E latency tails and mean runtime while preserving deterministic replay fields, fail-closed nn_state behavior, and existing differential parity posture.
