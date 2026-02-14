# Conformance Fixtures

This folder stores normalized oracle-vs-target fixtures for `ft-conformance`.

- `smoke_case.json`: minimal bootstrap fixture ensuring harness wiring works.
- `scalar_autograd_cases.json`: deterministic scalar DAC fixture family (strict + hardened).
- `tensor_meta_cases.json`: tensor metadata/indexing/alias invariants (contiguous, strided, scalar-offset, and adversarial fail-closed) for packet `FT-P2C-001`.
- `dispatch_key_cases.json`: dispatch key routing + mode-split fallback contract.
- `autograd_scheduler_cases.json`: deterministic scheduler ordering + reentrant policy contract.
- `serialization_cases.json`: checkpoint encode/decode + RaptorQ sidecar/proof contract.

Related adversarial/fuzz manifest (versioned):
- `artifacts/phase2c/ADVERSARIAL_FUZZ_CORPUS_MANIFEST_V1.json`

Related user-workflow scenario corpus artifacts:
- `artifacts/phase2c/USER_WORKFLOW_SCENARIO_CORPUS_V1.json`
- `artifacts/phase2c/USER_WORKFLOW_SCENARIO_GAP_LEDGER_V1.md`
- `artifacts/phase2c/e2e_forensics/golden_journey_coverage_v1.json`
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`

Related forensics UX/index artifacts:
- `artifacts/phase2c/FAILURE_FORENSICS_ENVELOPE_SCHEMA_V1.md`
- `artifacts/phase2c/e2e_forensics/crash_triage_full_v1.json`
- `artifacts/phase2c/e2e_forensics/failure_forensics_index_v1.json`

Related reliability budget gate artifacts:
- `artifacts/phase2c/RELIABILITY_BUDGET_POLICY_V1.json`
- `artifacts/phase2c/RELIABILITY_GATE_WORKFLOW_V1.md`
- `artifacts/phase2c/e2e_forensics/reliability_gate_report_v1.json`
