# FT-P2C-007 — Rust Implementation Plan + Module Boundary Skeleton

Packet: Device guard and backend transitions  
Subtask: `bd-3v0.18.4`

## 1) Module Boundary Justification

| Crate/module | Ownership | Why this boundary is required | Integration seam |
|---|---|---|---|
| `ft-device::DeviceGuard` + `ensure_same_device` | packet-local device compatibility contract | isolates device-guard semantics from dispatch routing logic | consumed by dispatch precondition checks and conformance mismatch fixtures |
| `ft-dispatch::DispatchKeySet` validation (`validate_for_scalar_binary`) | keyset fail-closed boundary (`EmptySet`, `NoTypeKey`, `NoBackendKey`, incompatible sets) | centralizes strict parser/state rejection to prevent drift between call sites | invoked by direct dispatch APIs and fixture-driven conformance paths |
| `ft-dispatch::dispatch_scalar_binary_with_keyset` | strict/hardened mode split and backend transition policy | makes composite/autograd fallback behavior explicit and auditable | emits deterministic decision tuple consumed by conformance/logging |
| `ft-dispatch` property-test surface | deterministic contract checks for mode split and keyset invariants | provides seed-stable behavior proofs before differential/e2e expansion | writes contract-compatible structured logs consumed by reliability tooling |
| `ft-conformance` dispatch suite + fixture set (`dispatch_key_cases.json`) | packet-level strict/hardened parity evidence and adversarial scenario execution | single source for packet route/output/error parity assertions | produces differential slices, e2e forensics logs, and gate-ready artifacts |
| `ft-conformance` forensics binaries (`run_e2e_matrix`, `triage_forensics_failures`, `build_failure_forensics_index`, `check_reliability_budgets`) | deterministic replay envelope and incident triage/index outputs | keeps packet closure evidence machine-ingestible and reproducible | consumes packet scenario logs and emits audit artifacts for closure beads |

## 2) Low-Risk Implementation Sequence (One Optimization Lever per Step)

| Step | Change scope | Semantic risk strategy | Single optimization lever |
|---|---|---|---|
| `S1` | lock device-guard + keyset validation fail-closed behavior | verify deterministic error taxonomy before widening fallback behavior | precompute keyset bitmask once per dispatch invocation |
| `S2` | enforce strict/hardened backend-transition policy for composite/autograd paths | gate mode-split behavior with deterministic decision evidence before coverage expansion | fast-path direct CPU routes to skip fallback branch checks |
| `S3` | expand unit/property coverage and contract-compatible structured logs | ensure replay fields are complete before differential packet wiring | reuse deterministic seed/fingerprint helper across tests |
| `S4` | wire differential/metamorphic/adversarial packet checks and reconciliation | block non-allowlisted hardened drift before e2e closure | packet-filtered comparator execution to reduce drift-noise runtime |
| `S5` | emit packet-scoped e2e replay/forensics artifacts and reliability gate outputs | require one-command replay + triage/index artifacts before packet closure | packet-filtered JSONL emission to minimize forensics processing overhead |

## 3) Detailed Test Implementation Plan

### 3.1 Unit/property suite plan

- `ft-device`
  - `guard_accepts_matching_device`
  - `same_device_check_returns_cpu`
  - planned in `bd-3v0.18.5`: guard/device mismatch fail-closed assertions for deterministic reason mapping
- `ft-dispatch`
  - `validate_requires_cpu_for_autograd`
  - `strict_mode_rejects_composite_fallback`
  - `hardened_mode_allows_composite_fallback`
  - `unknown_bits_fail_closed`
  - `prop_validate_requires_cpu_for_autograd`
  - `prop_mode_split_for_composite_keysets`
- `ft-conformance`
  - `strict_dispatch_conformance_is_green`
  - `hardened_dispatch_conformance_is_green`

### 3.2 Differential/metamorphic/adversarial hooks

- differential source artifact:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
- packet-slice artifacts (target in `bd-3v0.18.6`):
  - `artifacts/phase2c/FT-P2C-007/differential_packet_report_v1.json`
  - `artifacts/phase2c/FT-P2C-007/differential_reconciliation_v1.md`
- adversarial targets:
  - composite fallback escalation (`composite_route_mode_split`)
  - autograd-without-backend rejection (`autograd_without_cpu_fail_closed`)
  - malformed keyset families (`empty_keyset_fail_closed`, `unknown_dispatch_key_fail_closed`)
  - dtype/device mismatch fail-closed (`dtype_mismatch_fail_closed`, `device_mismatch_fail_closed`)
- deterministic threat seeds and abuse mapping are anchored in:
  - `artifacts/phase2c/FT-P2C-007/threat_model.md`

### 3.3 E2E script plan

- packet-scoped e2e command (target in `bd-3v0.18.7`):
  - `rch exec -- cargo run -p ft-conformance --bin run_e2e_matrix -- --mode both --packet FT-P2C-007 --output artifacts/phase2c/e2e_forensics/ft-p2c-007.jsonl --print-full-log`
- packet triage/index commands (target in `bd-3v0.18.7`):
  - `rch exec -- cargo run -p ft-conformance --bin triage_forensics_failures -- --input artifacts/phase2c/e2e_forensics/ft-p2c-007.jsonl --output artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_007_v1.json --packet FT-P2C-007`
  - `rch exec -- cargo run -p ft-conformance --bin build_failure_forensics_index -- --e2e artifacts/phase2c/e2e_forensics/ft-p2c-007.jsonl --triage artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_007_v1.json --output artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_007_v1.json`

## 4) Structured Logging Instrumentation Points

Required packet fields:
- `suite_id`
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `reason_code`
- `outcome`

Dispatch/device-transition additions:
- `dispatch_key`
- `backend_key`
- `selected_kernel`
- `keyset_bits`
- `fallback_path`
- `device_pair`
- `dtype_pair`
- `error_message`
- `contract_ids`

## 5) Conformance + Benchmark Integration Hooks

Conformance hooks:
- strict/hardened dispatch conformance suites
- packet differential report + reconciliation artifacts
- packet e2e replay + triage + failure-index outputs
- reliability budget policy checks (`check_reliability_budgets`)
- packet/global validator integration (`validate_phase2c_artifacts`)

Benchmark hooks:
- dispatch keyset-validation latency (p50/p95/p99) across representative packet keysets
- strict-vs-hardened transition overhead on composite-path probes
- device-guard compatibility check overhead on mismatch-heavy adversarial corpus
- fallback branch frequency tracking under packet fixture distributions

## 6) N/A Cross-Cutting Validation Note

This implementation-plan artifact is docs/planning for subtask D (`bd-3v0.18.4`).
Execution evidence ownership is explicitly delegated to:
- `bd-3v0.18.5` (unit/property + structured logs)
- `bd-3v0.18.6` (differential/metamorphic/adversarial)
- `bd-3v0.18.7` (e2e replay/forensics logging)
- `bd-3v0.18.8` (optimization/isomorphism)
- `bd-3v0.18.9` (final evidence pack)
