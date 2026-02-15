# FT-P2C-006 — Rust Implementation Plan + Module Boundary Skeleton

Packet: Serialization/checkpoint contract  
Subtask: `bd-3v0.17.4`

## 1) Module Boundary Justification

| Crate/module | Ownership | Why this boundary is required | Integration seam |
|---|---|---|---|
| `ft-serialize::CheckpointEnvelope` | checkpoint schema + canonical hashing boundary | centralizes strict/hardened decode policy and checksum/version fail-closed gates | consumed by conformance harness and packet parity artifacts |
| `ft-serialize::generate_raptorq_sidecar` | durability sidecar/proof generation | isolates RaptorQ-specific logic from checkpoint schema logic | called by conformance packet flow + durability pipeline binaries |
| `ft-runtime::DurabilityEnvelope` | runtime-level durability evidence model | separates runtime evidence semantics from wire-format serialization mechanics | consumed by API/runtime layers and long-lived artifact policy |
| `ft-conformance` serialization suite | fixture execution + parity policy checks | packet-level comparator for strict/hardened behavior and sidecar/proof determinism | emits packet parity reports and e2e forensic logs |
| `ft-conformance` forensic binaries (`run_e2e_matrix`, `triage_forensics_failures`, `build_failure_forensics_index`, `check_reliability_budgets`) | replay/failure envelope + reliability gating | keeps release gating and diagnostics deterministic and machine-ingestible | consumes packet fixtures/logs and produces closure artifacts |
| `artifacts/phase2c/FT-P2C-006/*` | packet contract/risk/threat/evidence docs | provides auditable mapping from behavior contracts to verification outputs | consumed by `validate_phase2c_artifacts` and final packet closure gates |

## 2) Low-Risk Implementation Sequence (One Optimization Lever per Step)

| Step | Change scope | Semantic risk strategy | Single optimization lever |
|---|---|---|---|
| `S1` | checkpoint schema normalization + deterministic hash boundary | prove canonicalization and fail-closed behavior before durability expansion | sort entries once at encode path and reuse normalized ordering |
| `S2` | strict/hardened decode path split + bounded hardened diagnostics | validate strict/hardened acceptance parity and allowlist compliance before fixture expansion | bounded diagnostic string truncation to cap hostile payload amplification |
| `S3` | RaptorQ sidecar/proof generation + decode integrity checks | require exact-byte recovery and deterministic proof hash before pipeline integration | symbol-size selection heuristic by payload size bucket |
| `S4` | conformance + differential/adversarial hooks for serialization packet | lock parity and threat-model coverage before performance work | reuse parsed fixture payload across strict/hardened executions |
| `S5` | packet e2e/replay/forensics + reliability gate integration | enforce deterministic crash triage/failure-index emission before closure | packet-filtered e2e emission to reduce forensics matrix runtime |

## 3) Detailed Test Implementation Plan

### 3.1 Unit/property suite plan

- `ft-serialize`
  - `checkpoint_round_trip_strict_works`
  - `strict_unknown_field_fail_closed`
  - `hardened_malformed_payload_returns_bounded_diagnostic`
  - `version_mismatch_is_fail_closed`
  - `checksum_mismatch_is_fail_closed`
  - `sidecar_generation_and_decode_proof_are_available`
  - `decode_proof_hash_is_deterministic`
- `ft-conformance`
  - `strict_serialization_conformance_is_green`
  - `hardened_serialization_conformance_is_green`

### 3.2 Differential/metamorphic/adversarial hooks

- differential source artifact:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
- adversarial targets (packet F ownership):
  - unknown field injection
  - schema version mismatch
  - checksum tamper
  - malformed JSON / incompatible top-level payload
  - RaptorQ corruption probe mismatch
- threat IDs and deterministic scenario seeds are anchored in:
  - `artifacts/phase2c/FT-P2C-006/threat_model.md`

### 3.3 E2E script plan

- packet-scoped e2e command:
  - `rch exec -- cargo run -p ft-conformance --bin run_e2e_matrix -- --mode both --output artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl`
  - `rg '"packet_id":"FT-P2C-006"' artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl > artifacts/phase2c/e2e_forensics/ft-p2c-006.jsonl`
- reliability/triage commands (packet slice):
  - `rch exec -- cargo run -p ft-conformance --bin triage_forensics_failures -- --input artifacts/phase2c/e2e_forensics/ft-p2c-006.jsonl --output artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_006_v1.json --packet FT-P2C-006`
  - `rch exec -- cargo run -p ft-conformance --bin build_failure_forensics_index -- --e2e artifacts/phase2c/e2e_forensics/ft-p2c-006.jsonl --triage artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_006_v1.json --output artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_006_v1.json`

## 4) Structured Logging Instrumentation Points

Required packet fields:
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `reason_code`

Serialization/durability additions:
- `source_hash`
- `schema_version`
- `repair_symbol_count`
- `constraints_symbol_count`
- `proof_hash_hex`
- `recovered_bytes`

## 5) Conformance + Benchmark Integration Hooks

Conformance hooks:
- strict/hardened serialization conformance suites
- packet-filtered e2e forensics slice
- crash triage + failure forensics index generation
- reliability budget gate (`check_reliability_budgets`)
- packet/global durability validation (`run_raptorq_durability_pipeline`, `validate_phase2c_artifacts`)

Benchmark hooks:
- serialization encode/decode p50/p95/p99 latency under packet fixture corpus
- sidecar generation overhead by payload size bucket
- recovery verification timing under deterministic corruption-probe drills

## 6) N/A Cross-Cutting Validation Note

This implementation-plan artifact is docs/planning only for subtask D.
Execution evidence is deferred to:
- `bd-3v0.17.5` (unit/property with structured logs)
- `bd-3v0.17.6` (differential/metamorphic/adversarial)
- `bd-3v0.17.7` (e2e replay/forensics logging)
