# FEATURE_PARITY

## Status Legend

- `not_started`
- `in_progress`
- `parity_green`
- `parity_gap`

## Parity Matrix

| Feature Family | Status | Notes |
|---|---|---|
| Tensor core semantics | in_progress | `FT-P2C-001` scalar metadata/value/version slice shipped |
| Dispatch key routing | parity_green | `FT-P2C-002` keyset model + strict/hardened mode split conformance |
| Autograd correctness | parity_green | `FT-P2C-004` dependency-driven scheduler + deterministic replay telemetry |
| CPU kernel subset | in_progress | scalar `add`/`mul` kernels green; tensor ops pending |
| Checkpoint compatibility | parity_green | `FT-P2C-006` typed checkpoint + fail-closed decode + RaptorQ sidecars |

## Current Green Scope

- `crates/ft-conformance/fixtures/scalar_autograd_cases.json`
- `crates/ft-conformance/fixtures/dispatch_key_cases.json`
- `crates/ft-conformance/fixtures/autograd_scheduler_cases.json`
- `crates/ft-conformance/fixtures/serialization_cases.json`

Modes tested for all listed families: strict + hardened.

## Required Evidence Per Feature Family

1. Differential fixture report.
2. Edge-case/adversarial test results.
3. Benchmark delta (when performance-sensitive).
4. Documented compatibility exceptions (if any).
5. RaptorQ sidecar + decode-proof chain for durable parity bundles.
