# FT-P2C-006 E2E Replay + Forensics Linkage (v1)

## Scope

- e2e log: `artifacts/phase2c/e2e_forensics/ft-p2c-006.jsonl`
- triage summary: `artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_006_v1.json`
- packet failure index: `artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_006_v1.json`

## Outcomes

- packet filter recorded: `FT-P2C-006`
- total entries: `4` (strict=2, hardened=2)
- failed entries: `0`
- triaged incidents: `0`

Suites exercised:
- `serialization`

## Deterministic Replay Contract Status

Required fields are present for every log entry:
- `scenario_id`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `reason_code`

## Cross-Evidence Linkage

- unit/property evidence commands are represented in packet failure-index templates and serialization conformance runs.
- differential linkage references:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
  - `artifacts/phase2c/FT-P2C-006/differential_packet_report_v1.json`
  - `artifacts/phase2c/FT-P2C-006/differential_reconciliation_v1.md`
- optimization linkage references:
  - `artifacts/phase2c/FT-P2C-006/optimization_delta_v1.json`
  - `artifacts/phase2c/FT-P2C-006/optimization_isomorphism_v1.md`
- risk linkage: `artifacts/phase2c/FT-P2C-006/risk_note.md`

## Method-Stack Note

- alien-artifact-coding: replay + forensics linkage emitted as deterministic packet-scoped artifacts.
- extreme-software-optimization: serialization sidecar cache lever has explicit isomorphism evidence and retained determinism checks.
- RaptorQ-everywhere durability: packet parity sidecar/decode artifacts remain the durability anchor for this packet.
- security/compatibility doctrine: packet-filtered triage confirms strict/hardened serialization behavior with no unclassified failures.
