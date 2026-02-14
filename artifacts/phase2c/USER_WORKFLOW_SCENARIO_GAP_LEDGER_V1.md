# User Workflow Scenario Gap Ledger V1

Bead: `bd-3v0.20`  
Date: 2026-02-14

## Covered Journey Classes
- `golden_happy_path`: scalar autograd + tensor metadata indexing
- `golden_regression_path`: dispatch route parity, autograd scheduler determinism, serialization replay
- `malformed_input_path`: tensor metadata rank/stride mismatch + storage overflow
- `degraded_environment_path`: strict vs hardened dispatch composite fallback split

## Explicit Non-Covered Branches (Follow-up Required)

| Gap ID | Non-covered branch | Why not fully covered yet | Follow-up beads | Owner note |
|---|---|---|---|---|
| `GW-001` | device mismatch + tensor compatibility negative scenarios (`DeviceError::Mismatch`, `TensorCompatError`) | no dedicated conformance fixture/e2e scenario yet | `bd-3v0.23.10`, `bd-3v0.12.6`, `bd-3v0.20` | add fixture rows and packet-scoped e2e slice for mismatch reason codes |
| `GW-002` | legacy oracle timeout/cancel/hang envelope | oracle unavailability is tracked, but timeout kill-path scenario is not yet modeled | `bd-3v0.21`, `bd-3v0.23.10` | extend forensics UX and add timeout reason taxonomy |
| `GW-003` | runtime durability-ledger assertions in integrated conformance/e2e path | sidecar generation is covered, but runtime durability evidence is not asserted in conformance logs | `bd-3v0.9`, `bd-3v0.10`, `bd-3v0.17.7` | bridge `ft-runtime` durability entries into conformance/e2e artifacts |
| `GW-004` | explicit adversarial serialization incompatible-payload fixture | strict/hash/version malformed cases are covered, but incompatible payload branch lacks dedicated journey entry | `bd-3v0.17.6`, `bd-3v0.17.7` | add new fixture case and journey row in corpus V2 |

## Replay and Forensics Contract Reminder
- Scenario IDs must remain stable by `scenario_id` contract in `crates/ft-conformance/src/lib.rs:2345`.
- Every journey evidence run must include `scenario_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `replay_command`, and `reason_code` (`crates/ft-conformance/src/logging.rs:11`).
- Gap closure updates must append new journey IDs to `artifacts/phase2c/USER_WORKFLOW_SCENARIO_CORPUS_V1.json` or publish `V2` if schema changes.
