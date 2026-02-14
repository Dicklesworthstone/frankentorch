# CRASH_TRIAGE_WORKFLOW_V1

Schema: `ft-crash-triage-workflow-v1`  
Bead: `bd-3v0.7`

## 1. Scope

This workflow classifies failed forensic log entries into deterministic crash classes, emits replay-ready incident artifacts, and routes ownership for remediation.

Input log schema:
- `crates/ft-conformance/src/logging.rs` (`StructuredCaseLog`)

Classifier binary:
- `cargo run -p ft-conformance --bin triage_forensics_failures -- --input <jsonl> --output <json>`

## 2. Deterministic triage pipeline

1. Generate forensic log matrix:
```bash
cargo run -p ft-conformance --bin run_e2e_matrix -- --mode both --output artifacts/phase2c/e2e_forensics/e2e_matrix.jsonl
```
2. Produce crash triage report:
```bash
cargo run -p ft-conformance --bin triage_forensics_failures -- --input artifacts/phase2c/e2e_forensics/e2e_matrix.jsonl --output artifacts/phase2c/e2e_forensics/crash_triage_v1.json
```
3. Optional packet-focused triage:
```bash
cargo run -p ft-conformance --bin triage_forensics_failures -- --input artifacts/phase2c/e2e_forensics/e2e_matrix.jsonl --packet FT-P2C-004 --output artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_004.json
```
4. Route incidents by `owner_hint` and track with bead links.

## 3. Classification map

| Class | Severity | Owner hint | Typical reason-code patterns |
|---|---|---|---|
| `autograd_state` | `critical` | `ft-autograd-owners` | `scheduler`, `reentrant`, `dependency`, `unknown_node` |
| `dispatch_routing` | `high` | `ft-dispatch-owners` | `dispatch`, `keyset`, `fallback` |
| `serialization_parser` | `high` | `ft-serialize-and-durability-owners` | `serialization`, `decode`, `checksum`, `unknown_field`, `invalid_json` |
| `tensor_meta_state` | `high` | `ft-core-owners` | `tensor_meta`, `rank`, `index`, `stride` |
| `oracle_infra` | `medium` | `ft-conformance-infra-owners` | `oracle` |
| `unclassified` | `high` | `crash-triage-owner` | all unmatched reason codes |

## 4. Incident artifact contract

Triage output schema version:
- `ft-crash-triage-v1`

Required incident fields:
- `incident_id`
- `class`
- `severity`
- `owner_hint`
- `packet_id`
- `suite_id`
- `scenario_id`
- `mode`
- `reason_code`
- `replay_command`
- `artifact_refs`
- `seed`
- `env_fingerprint`
- `occurrences`
- `last_seen_unix_ms`

## 5. Forensic replay requirements

Every routed incident must keep:
- original `replay_command`
- `artifact_refs` intact
- packet + mode labels
- deterministic `seed`

If incident class is `critical`:
- escalate immediately
- attach linked bead ID and parity/e2e artifact paths
- require one reproducible replay transcript before closure

## 6. Failure-injection hooks (current)

- Strict-mode fail-closed dispatch paths from `dispatch_key_cases.json`
- Reentrant depth overflow strict/hardened branches in scheduler flow
- Serialization malformed/compatibility drift paths via strict/hardened decode logic
- Oracle-availability branch in differential conformance (classified as `oracle_infra`)

## 7. Known gaps and follow-up

- No timeout/kill envelope yet for legacy oracle subprocess hangs.
- No dedicated flake-vs-deterministic classifier branch yet.
- No automatic ownership ticket creation in this pass (manual routing remains).

Follow-up beads:
- `bd-3v0.23.8`
- `bd-3v0.20`
- `bd-3v0.23.10`
