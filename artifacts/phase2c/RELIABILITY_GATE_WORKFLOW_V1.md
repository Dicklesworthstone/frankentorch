# Reliability Gate Workflow V1

Bead: `bd-3v0.22`  
Policy: `artifacts/phase2c/RELIABILITY_BUDGET_POLICY_V1.json`

## Goal

Fail CI with precise, actionable diagnostics when reliability budgets are violated across coverage, pass ratio, or flake ceilings.

## Gate Inputs

- E2E forensic log stream:
  - `artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl`
- Reliability policy:
  - `artifacts/phase2c/RELIABILITY_BUDGET_POLICY_V1.json`
- Optional index artifacts for triage handoff:
  - `artifacts/phase2c/e2e_forensics/crash_triage_full_v1.json`
  - `artifacts/phase2c/e2e_forensics/failure_forensics_index_v1.json`

## Gate Execution

- Run checker:
  - `cargo run -p ft-conformance --bin check_reliability_budgets -- --policy artifacts/phase2c/RELIABILITY_BUDGET_POLICY_V1.json --e2e artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl --output artifacts/phase2c/e2e_forensics/reliability_gate_report_v1.json`

## Budget Families

1. Coverage floors:
   - Per packet minimum scenario count and required suite presence.
2. Pass ratio:
   - Per packet pass ratio threshold.
3. Global failure ceiling:
   - Maximum failed e2e entries.
4. Flake ceiling:
   - Maximum scenario IDs that show conflicting outcomes in one window.
5. Reason taxonomy hygiene:
   - Unknown reason codes must stay within budget.

## Flake Policy

Detection rule:
- same `scenario_id` appears with both pass and non-pass outcomes in one gate window.

Quarantine and retry policy:
- retry up to 2 times for suspected flakes.
- if still unstable, tag with `flake-quarantine` and open/update bead linked to `bd-3v0.22`.
- include `scenario_id`, `reason_code`, replay command, and artifact references in the bead update.
- de-quarantine after 3 consecutive clean runs.

## Required CI Output Shape

When budgets fail, report must include:
- exact budget ID(s)
- failing packet/suite/scenario IDs
- linked artifact refs (`e2e`, `crosswalk`, `forensics index`)
- remediation hint from policy

This ensures no hidden manual steps are required for triage.
