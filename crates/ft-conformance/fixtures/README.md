# Conformance Fixtures

This folder stores normalized oracle-vs-target fixtures for `ft-conformance`.

- `smoke_case.json`: minimal bootstrap fixture ensuring harness wiring works.
- `scalar_autograd_cases.json`: deterministic scalar DAC fixture family (strict + hardened).
- `tensor_meta_cases.json`: tensor metadata/indexing/alias invariants (contiguous, strided, scalar-offset, and adversarial fail-closed) for packet `FT-P2C-001`.
- `dispatch_key_cases.json`: dispatch key routing + mode-split fallback contract.
- `autograd_scheduler_cases.json`: deterministic scheduler ordering + reentrant policy contract.
- `serialization_cases.json`: checkpoint encode/decode + RaptorQ sidecar/proof contract.
