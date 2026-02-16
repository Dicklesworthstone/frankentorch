# FT-P2C-007 E2E Replay + Forensics Linkage (v1)

## Scope

- e2e log: `artifacts/phase2c/e2e_forensics/ft-p2c-007.jsonl`
- triage summary: `artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_007_v1.json`
- packet failure index: `artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_007_v1.json`

## Outcomes

- packet filter recorded: `FT-P2C-007`
- total entries: `20` (strict=10, hardened=10)
- failed entries: `0`
- triaged incidents: `0`

Suites exercised:
- `dispatch_key`

## Deterministic Replay Contract Status

Required fields are present for every log entry:
- `scenario_id`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `reason_code`

## Cross-Evidence Linkage

- unit/property evidence commands are embedded in the packet linkage summary and packet failure-index templates.
- differential linkage references:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
  - `artifacts/phase2c/FT-P2C-007/differential_packet_report_v1.json`
  - `artifacts/phase2c/FT-P2C-007/differential_reconciliation_v1.md`
- optimization linkage references:
  - `artifacts/phase2c/FT-P2C-007/optimization_delta_v1.json`
  - `artifacts/phase2c/FT-P2C-007/optimization_isomorphism_v1.md`

## Method-Stack Note

- alien-artifact-coding: packet-007 replay/forensics linkage now emits deterministic dispatch/device-guard traces with packet projection metadata.
- extreme-software-optimization: this bead introduces no performance lever; behavior-isomorphism is preserved with zero packet e2e failures.
- RaptorQ-everywhere durability: parity sidecar/decode artifacts remain anchored under packet-level closure beads.
- security/compatibility doctrine: strict/hardened dispatch_key scenarios remain fail-closed except explicit hardened composite fallback allowlisting.
