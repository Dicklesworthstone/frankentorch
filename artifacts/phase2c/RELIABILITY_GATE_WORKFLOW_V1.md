# Reliability Gate Workflow V1

Beads: `bd-3v0.10`, `bd-3v0.22`  
Policy: `artifacts/phase2c/RELIABILITY_BUDGET_POLICY_V1.json`

CI workflow implementation: `.github/workflows/phase2c_reliability_gates.yml`

## Goal

Fail CI with precise, actionable diagnostics when Phase-2C reliability budgets or gate contracts are violated.

## Gate Inputs

- E2E forensic log streams:
  - `artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl`
  - `artifacts/phase2c/e2e_forensics/e2e_matrix_gate_window_v1.jsonl`
  - `artifacts/phase2c/e2e_forensics/ft-p2c-003.jsonl`
  - `artifacts/phase2c/e2e_forensics/ft-p2c-005.jsonl`
  - `artifacts/phase2c/e2e_forensics/ft-p2c-007.jsonl`
  - `artifacts/phase2c/e2e_forensics/ft-p2c-008.jsonl`
- Reliability policy:
  - `artifacts/phase2c/RELIABILITY_BUDGET_POLICY_V1.json`
- Optional index artifacts for triage handoff:
  - `artifacts/phase2c/e2e_forensics/crash_triage_full_v1.json`
  - `artifacts/phase2c/e2e_forensics/failure_forensics_index_v1.json`

## G1..G8 Gate Topology

| Gate | Contract | Primary command(s) | Blocking evidence artifacts |
|---|---|---|---|
| `G1` | fmt/lint hygiene | `rch exec -- env CARGO_TARGET_DIR=target_phase2c cargo fmt --check` + `rch exec -- env CARGO_TARGET_DIR=target_phase2c cargo clippy --workspace --all-targets -- -D warnings` | clippy/fmt output in CI logs |
| `G2` | unit/property + structured log contract | `rch exec -- env CARGO_TARGET_DIR=target_phase2c cargo test --workspace` | packet `unit_property_quality_report_v1.json` + `UNIT_E2E_LOGGING_CROSSWALK_V1.json` |
| `G3` | differential/metamorphic/adversarial parity | `rch exec -- env CARGO_TARGET_DIR=target_phase2c cargo run -p ft-conformance --bin run_differential_report -- --mode both --output artifacts/phase2c/conformance/differential_report_v1.json` | packet `differential_packet_report_v1.json` + reconciliation notes |
| `G4` | e2e replay + forensics completeness | `rch exec -- env CARGO_TARGET_DIR=target_phase2c cargo run -p ft-conformance --bin run_e2e_matrix -- --mode both --output artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl` plus packet slices (`FT-P2C-003/005/007/008`) merged into `e2e_matrix_gate_window_v1.jsonl` | packet `ft-p2c-*.jsonl`, merged gate-window log, crash triage, forensics index |
| `G5` | perf tails + isomorphism | packet microbench tests under `ft-conformance` via `rch exec -- env CARGO_TARGET_DIR=target_phase2c cargo test -p ft-conformance packet_e2e_microbench -- --nocapture` | packet `optimization_delta_v1.json` + `optimization_isomorphism_v1.md` |
| `G6` | artifact schema + packet lock validation | `rch exec -- env CARGO_TARGET_DIR=target_phase2c cargo run -p ft-conformance --bin validate_phase2c_artifacts -- /data/projects/frankentorch` | validator summary (`ok=true`), packet required artifacts present |
| `G7` | RaptorQ decode/integrity durability | `rch exec -- env CARGO_TARGET_DIR=target_phase2c cargo run -p ft-conformance --bin run_raptorq_durability_pipeline` | `RAPTORQ_REPAIR_SYMBOL_MANIFEST_V1.json`, `RAPTORQ_INTEGRITY_SCRUB_REPORT_V1.json`, `RAPTORQ_DECODE_PROOF_EVENTS_V1.json` |
| `G8` | readiness sign-off + residual risk | packet closure + readiness drill (`bd-3v0.11`) | `artifacts/phase2c/READINESS_DRILL_SIGNOFF_V1.md`, residual risk and allowlist deltas |

`G8` enforcement mode:
- default CI runs keep `G8` declared but deferred.
- run `workflow_dispatch` with `enforce_g8=true` to make `G8` blocking once readiness drill artifacts exist.

## Reliability Budget Execution

- Run reliability checker:
  - `rch exec -- env CARGO_TARGET_DIR=target_phase2c cargo run -p ft-conformance --bin check_reliability_budgets -- --policy artifacts/phase2c/RELIABILITY_BUDGET_POLICY_V1.json --e2e artifacts/phase2c/e2e_forensics/e2e_matrix_gate_window_v1.jsonl --output artifacts/phase2c/e2e_forensics/reliability_gate_report_v1.json`

Gate-window assembly rule:
- compose `e2e_matrix_gate_window_v1.jsonl` from `e2e_matrix_full_v1.jsonl` plus packet slices (`FT-P2C-003/005/007/008`) and de-duplicate identical JSONL rows before running `check_reliability_budgets`.

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

## Required CI Failure Forensics Shape

When any gate fails, failure output must include:
- exact gate ID (`G1`..`G8`) and budget ID(s)
- failing packet/suite/scenario IDs
- deterministic replay command(s)
- `artifact_refs` entries for e2e logs, differential slices, and forensics index
- `reason_code` and remediation hint from policy

This ensures no hidden manual steps are required for triage.
