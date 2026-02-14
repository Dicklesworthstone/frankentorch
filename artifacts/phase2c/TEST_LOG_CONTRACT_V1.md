# TEST_LOG_CONTRACT_V1

Schema version: `ft-conformance-log-v1`

Purpose: canonical unit/property/e2e structured log contract for deterministic replay, differential triage, and forensic correlation.

## Required Fields

Every test/e2e log entry MUST include:

- `schema_version`
- `ts_unix_ms`
- `suite_id`
- `scenario_id`
- `fixture_id`
- `packet_id`
- `mode` (`strict` or `hardened`)
- `seed`
- `env_fingerprint`
- `artifact_refs` (non-empty)
- `replay_command`
- `outcome` (`pass` or `fail`)
- `reason_code`

## Determinism Rules

- `scenario_id` must be stable for `(suite_id, mode, case_name)`.
- `seed` must be deterministic from stable scenario components.
- `env_fingerprint` must be deterministic from runtime-invariant environment signals.
- `artifact_refs` must point to fixture/parity artifacts needed for triage.

## Replay Rule

`replay_command` must be executable without ad-hoc debug instrumentation and must reproduce the failing scenario class in the same mode.

## E2E Correlation Rule

`scenario_id` is the primary join key across:

- unit/property logs
- differential/metamorphic/adversarial logs
- e2e forensics logs

`packet_id` and `artifact_refs` provide packet-scoped audit traceability.

## Current Emitters

- `run_scalar_conformance`
- `run_dispatch_conformance`
- `run_autograd_scheduler_conformance`
- `run_serialization_conformance`
- `run_e2e_matrix` (JSONL emitter)

## Packet-Scoped E2E Filter

The e2e emitter supports packet-scoped filtering (`--packet FT-P2C-00X`) while preserving the same schema. This enables packet-local replay and CI slicing without changing forensic field contracts.
