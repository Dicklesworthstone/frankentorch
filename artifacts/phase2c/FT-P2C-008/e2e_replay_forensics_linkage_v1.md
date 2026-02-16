# FT-P2C-008 E2E Replay + Forensics Linkage (v1)

## Scope

- e2e log: `artifacts/phase2c/e2e_forensics/ft-p2c-008.jsonl`
- triage summary: `artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_008_v1.json`
- packet failure index: `artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_008_v1.json`

## Outcomes

- packet filter recorded: `FT-P2C-008`
- total entries: `20` (strict=10, hardened=10)
- failed entries: `0`
- triaged incidents: `0`

Suites exercised:
- `nn_state`

## Deterministic Replay Contract Status

Required fields are present for every log entry:
- `scenario_id`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `reason_code`

## Cross-Evidence Linkage

- unit/property evidence commands are embedded in the packet linkage summary for strict/hardened nn_state conformance and packet-008 property checks.
- differential linkage references:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
  - `artifacts/phase2c/FT-P2C-008/differential_packet_report_v1.json`
  - `artifacts/phase2c/FT-P2C-008/differential_reconciliation_v1.md`
- optimization linkage references:
  - `artifacts/phase2c/FT-P2C-008/optimization_delta_v1.json`
  - `artifacts/phase2c/FT-P2C-008/optimization_isomorphism_v1.md`

## Method-Stack Note

- alien-artifact-coding: packet-008 replay/forensics linkage now emits deterministic nn_state traces across strict+hardened scenario families.
- extreme-software-optimization: no optimization lever changed in this bead; behavior-isomorphism remains governed by differential + e2e pass envelopes.
- RaptorQ-everywhere durability: sidecar/decode closure is deferred to the packet final evidence bead.
- security/compatibility doctrine: strict/hardened nn_state scenarios are fully replayable with zero packet-filtered failures.
