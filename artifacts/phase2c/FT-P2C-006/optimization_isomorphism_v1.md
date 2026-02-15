# FT-P2C-006 Optimization + Isomorphism Evidence (v1)

## Optimization Lever

- ID: `serialization-sidecar-cache-with-uncached-determinism-check`
- Change: cache successful sidecar/proof results keyed by payload+repair budget and keep determinism validation strict by forcing the second proof generation call to bypass cache.
- Path:
  - `crates/ft-conformance/src/lib.rs`

## Benchmark Delta (`packet_e2e_microbench_serialization_produces_percentiles`)

- Baseline (pre-optimization): `p50=23203512ns`, `p95=24940434ns`, `p99=24940434ns`, `mean=23712256ns`
- Post (after optimization): `p50=13893901ns`, `p95=16884031ns`, `p99=16884031ns`, `mean=15962527ns`
- Improvement: `p50=40.122% reduction`, `p95=32.303% reduction`, `p99=32.303% reduction`, `mean=32.682% reduction`

## Isomorphism Checks

- serialization conformance + packet checks: full `ft-conformance` suite passed (`rch exec -- cargo test -p ft-conformance -- --nocapture`)
- differential comparator presence: `differential_serialization_adds_metamorphic_and_adversarial_checks` passed
- e2e packet filter behavior: `e2e_matrix_packet_filter_includes_serialization_packet_entries` passed
- quality gates: workspace `cargo check`, workspace `cargo clippy -D warnings`, and `cargo fmt --check` passed

## Acceptance Note

This lever is accepted for FT-P2C-006 because it materially lowers packet e2e latency while preserving strict/hardened mode behavior, deterministic proof-hash contracts, and adversarial/differential serialization invariants.
