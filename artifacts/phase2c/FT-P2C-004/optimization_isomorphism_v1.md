# FT-P2C-004 Optimization + Isomorphism Evidence (v1)

## Optimization Lever

- ID: `autograd-scheduler-queue-capacity-compaction`
- Change: pre-size scheduler queue/trace buffers and remove a redundant reachable-node branch in the backward ready-queue loop.
- Paths:
  - `crates/ft-autograd/src/lib.rs`
  - `crates/ft-conformance/src/lib.rs` (packet-004 microbench hook)

## Benchmark Delta (`packet_e2e_microbench_autograd_scheduler_produces_percentiles`)

- Baseline (pre-optimization): `p50=296246ns`, `p95=416726ns`, `p99=416726ns`, `mean=335356ns`
- Post (after optimization): `p50=297897ns`, `p95=376948ns`, `p99=376948ns`, `mean=335703ns`
- Tail improvement: `p95=9.545% reduction`, `p99=9.545% reduction`
- Neutral/slight regression outside tails: `p50=-0.557%`, `mean=-0.103%`

## Isomorphism Checks

- unit/property: full `ft-autograd` test suite passed (`rch exec -- cargo test -p ft-autograd -- --nocapture`)
- differential comparator presence: `differential_autograd_scheduler_adds_metamorphic_and_adversarial_checks` passed
- e2e packet filter behavior: `e2e_matrix_packet_filter_includes_autograd_scheduler_packet_entries` passed
- quality gates: workspace `cargo check`, `cargo clippy -D warnings`, and `cargo fmt --check` passed

## Acceptance Note

This lever is accepted for FT-P2C-004 because it improves latency tails (p95/p99), which are the primary packet performance target, while preserving deterministic autograd and conformance invariants.
