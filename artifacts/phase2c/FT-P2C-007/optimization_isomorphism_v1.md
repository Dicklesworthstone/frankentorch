# FT-P2C-007 Optimization + Isomorphism Evidence (v1)

## Optimization Lever

- ID: `dispatch-packet-projection-fixture-cache-fastpath`
- Change: route packet-007 E2E through the fixture-cached dispatch projection path and benchmark it against the legacy emitter path to preserve deterministic replay while reducing packet latency tails.
- Path:
  - `crates/ft-conformance/src/lib.rs`

## Benchmark Delta (`packet_e2e_microbench_device_guard_legacy_vs_optimized_profiles`)

- Baseline (legacy emitter): `p50=1523698ns`, `p95=2041713ns`, `p99=2041713ns`, `mean=1660851ns`
- Post (optimized emitter): `p50=1465032ns`, `p95=1605280ns`, `p99=1605280ns`, `mean=1482731ns`
- Improvement: `p50=3.850% reduction`, `p95=21.376% reduction`, `p99=21.376% reduction`, `mean=10.725% reduction`

## Isomorphism Checks

- packet-specific e2e filter behavior: `e2e_matrix_packet_filter_includes_device_guard_packet_entries` passed
- packet-specific projection hygiene: `ft_p2c_007_projection_strips_shadowed_flatten_keys` passed
- packet-specific benchmark consistency: `packet_e2e_microbench_device_guard_produces_percentiles` and legacy-vs-optimized comparison passed
- dispatch conformance parity gates: `strict_dispatch_conformance_is_green` + `hardened_dispatch_conformance_is_green` passed
- differential packet evidence remains green via:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
  - `artifacts/phase2c/FT-P2C-007/differential_packet_report_v1.json`
  - `artifacts/phase2c/FT-P2C-007/differential_reconciliation_v1.md`

## Acceptance Note

This lever is accepted for FT-P2C-007 because it lowers packet-level strict/hardened E2E latency tails and mean runtime while preserving deterministic replay fields, fail-closed dispatch/device-guard behavior, and existing differential parity posture.
