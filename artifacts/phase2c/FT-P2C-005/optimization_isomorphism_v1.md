# FT-P2C-005 Optimization + Isomorphism Evidence (v1)

## Optimization Lever

- ID: `cpu-kernel-e2e-fixture-preload-with-shared-projection-execution`
- Change: preload scalar/tensor-meta/dispatch fixtures once per run and reuse them across strict+hardened execution while preserving FT-P2C-005 projection semantics.
- Path:
  - `crates/ft-conformance/src/lib.rs`

## Benchmark Delta (`packet_e2e_microbench_cpu_kernel_legacy_vs_optimized_profiles`)

- Baseline (legacy fixture-loading path): `p50=10282230ns`, `p95=14150815ns`, `p99=14150815ns`, `mean=10814785ns`
- Post (optimized preload path): `p50=9629848ns`, `p95=9885917ns`, `p99=9885917ns`, `mean=9232517ns`
- Improvement: `p50=6.345% reduction`, `p95=30.139% reduction`, `p99=30.139% reduction`, `mean=14.631% reduction`

## Isomorphism Checks

- packet regression + projection behavior:
  - `packet_e2e_microbench_cpu_kernel_legacy_vs_optimized_profiles`
  - `e2e_matrix_packet_filter_includes_cpu_kernel_packet_entries`
  - `e2e_matrix_unfiltered_keeps_ft_p2c_005_projection_entries`
- differential linkage remains unchanged:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
  - `artifacts/phase2c/FT-P2C-005/differential_packet_report_v1.json`
  - `artifacts/phase2c/FT-P2C-005/differential_reconciliation_v1.md`
- e2e replay linkage remains deterministic:
  - `artifacts/phase2c/e2e_forensics/ft-p2c-005.jsonl`
  - `artifacts/phase2c/FT-P2C-005/e2e_replay_forensics_linkage_v1.json`
- workspace quality gates:
  - `rch exec -- cargo check --workspace --all-targets`
  - `rch exec -- cargo clippy --workspace --all-targets -- -D warnings`
  - `rch exec -- cargo fmt --check`

## Memory/Profiling Note

Detailed profiling method and raw comparator output are recorded in:

- `artifacts/phase2c/FT-P2C-005/bd-3v0.16.8_profile_v1.md`
