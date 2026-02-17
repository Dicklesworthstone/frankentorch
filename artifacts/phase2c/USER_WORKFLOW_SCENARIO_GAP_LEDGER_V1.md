# User Workflow Scenario Gap Ledger V1

Bead: `bd-3v0.20`  
Date: 2026-02-14

## Covered Journey Classes
- `golden_happy_path`: scalar autograd + tensor metadata indexing
- `golden_regression_path`: dispatch route parity, autograd scheduler determinism, serialization replay
- `malformed_input_path`: tensor metadata rank/stride mismatch + storage overflow
- `degraded_environment_path`: strict vs hardened dispatch composite fallback split
- `adversarial_fail_closed_path`: FT-P2C-007 device/dtype mismatch dispatch rejection evidence
- `adversarial_fail_closed_path`: FT-P2C-006 incompatible serialization payload rejection evidence

## Explicit Branch Tracking (Open Follow-up + Recently Closed)

| Gap ID | Non-covered branch | Why not fully covered yet | Follow-up beads | Owner note |
|---|---|---|---|---|
| `GW-001` | device mismatch + tensor compatibility negative scenarios (`DeviceError::Mismatch`, `TensorCompatError`) | closed on 2026-02-17 via `UJ-FT-009` + `BHV-FT-007` packet-scoped scenario IDs and replay commands | `bd-3v0.23.10`, `bd-3v0.12.6`, `bd-3v0.20`, `bd-3qrd` | keep differential assertions bound to `INV-DTYPE-DEVICE-COMPAT` and `expected_error_observed` |
| `GW-002` | legacy oracle timeout/cancel/hang envelope | oracle unavailability is tracked, but timeout kill-path scenario is not yet modeled | `bd-3v0.21`, `bd-3v0.23.10` | extend forensics UX and add timeout reason taxonomy |
| `GW-003` | runtime durability-ledger assertions in integrated conformance/e2e path | sidecar generation is covered, but runtime durability evidence is not asserted in conformance logs | `bd-3v0.9`, `bd-3v0.10`, `bd-3v0.17.7` | bridge `ft-runtime` durability entries into conformance/e2e artifacts |
| `GW-004` | explicit adversarial serialization incompatible-payload fixture | closed on 2026-02-17 via `checkpoint_incompatible_payload_top_level_array` fixture + `UJ-FT-010` journey mapping | `bd-3v0.17.6`, `bd-3v0.17.7`, `bd-1wcb` | keep fail-closed reason-code checks bound to `expected_error_observed` |

## Replay and Forensics Contract Reminder
- Scenario IDs must remain stable by `scenario_id` contract in `crates/ft-conformance/src/lib.rs:2345`.
- Every journey evidence run must include `scenario_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `replay_command`, and `reason_code` (`crates/ft-conformance/src/logging.rs:11`).
- Gap closure updates must append new journey IDs to `artifacts/phase2c/USER_WORKFLOW_SCENARIO_CORPUS_V1.json` or publish `V2` if schema changes.
