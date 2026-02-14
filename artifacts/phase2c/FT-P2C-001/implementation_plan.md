# FT-P2C-001 — Rust Implementation Plan + Module Boundary Skeleton

Packet: Tensor metadata + storage core  
Subtask: `bd-3v0.12.4`

## 1) Module Boundary Justification

| Crate/module | Ownership | Why this boundary is required | Integration seam |
|---|---|---|---|
| `ft-core::TensorMeta` | metadata validation, stride/index math, fail-closed guards | keeps shape/stride/offset invariants local and auditable | consumed by `ft-dispatch`, `ft-autograd`, `ft-api` |
| `ft-core::ScalarTensor` | scalar storage/version/alias identity | centralizes version and alias semantics | produced/consumed by kernel + dispatch + autograd |
| `ft-device` | device compatibility guardrail | isolates device mismatch policy from compute paths | called before dispatch execution |
| `ft-kernel-cpu` | concrete scalar kernels (`add_scalar`, `mul_scalar`) | keeps arithmetic leaf behavior simple and testable | invoked only through dispatch |
| `ft-dispatch` | dispatch key validation and mode split | strict/hardened routing policy belongs in one place | returns `DispatchOutcome` with decision telemetry |
| `ft-autograd` | tape graph + deterministic backward scheduling | graph replay and dependency invariants require dedicated owner | uses dispatch outcomes and emits `BackwardReport` |
| `ft-conformance` | packet fixture orchestration + differential + e2e logs | centralized evidence production and replay contract | runs all packet suites and emits forensic artifacts |

## 2) Low-Risk Implementation Sequence (One Optimization Lever per Step)

| Step | Change scope | Semantic risk strategy | Single optimization lever |
|---|---|---|---|
| `S1` | finalize metadata/index fail-closed guards in `ft-core` | prove strict/hardened parity for invalid metadata cases first | contiguous/stride index helper reuse to avoid duplicate arithmetic |
| `S2` | finalize scalar kernel + dispatch wiring | keep strict route canonical, gate hardened fallback to explicit branch | avoid redundant dispatch-key recomputation per op |
| `S3` | stabilize backward scheduling telemetry and deterministic ordering | preserve dependency accounting before any optimization | preallocate queue capacity using reachable node count |
| `S4` | integrate conformance fixtures and differential checks for packet slice | lock replayability and drift classification before extension | canonical sort once at report build, not per check append |
| `S5` | wire e2e matrix + triage + failure index artifacts for packet scope | treat reliability gates as release blockers | stream JSONL writes in single pass to reduce IO churn |

## 3) Detailed Test Implementation Plan

### 3.1 Unit/property suite plan

- `ft-core`
  - `index_rank_and_bounds_are_guarded`
  - `custom_strides_validate_and_index_into_storage`
  - `alias_view_shares_storage_identity`
  - `out_of_place_result_gets_new_storage_and_version_bump`
- `ft-dispatch`
  - `unknown_bits_fail_closed`
  - `strict_mode_rejects_composite_fallback`
  - `hardened_mode_allows_composite_fallback`
- `ft-autograd`
  - `add_backward_matches_expected_gradient`
  - `mul_backward_matches_expected_gradient`
  - `dependency_scheduler_waits_for_all_children`

### 3.2 Differential/metamorphic/adversarial hooks

- differential artifact: `artifacts/phase2c/conformance/differential_report_v1.json`
- metamorphic tensor-meta offset-shift checks in `run_differential_conformance`
- adversarial fail-closed fixtures:
  - `tensor_meta/strict:invalid_rank_stride_mismatch`
  - `tensor_meta/strict:invalid_storage_offset_overflow`

### 3.3 E2E script plan

- packet-scoped e2e run command:
  - `cargo run -p ft-conformance --bin run_e2e_matrix -- --mode both --packet FT-P2C-001 --output artifacts/phase2c/e2e_forensics/ft-p2c-001.jsonl`
- replay commands are pulled from e2e log entries and passed through crash triage/index artifacts.

## 4) Structured Logging Instrumentation Points

Required fields for all packet events:
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `reason_code`

Instrumentation points:
- dispatch decision boundary (`selected_key`, `fallback_used`)
- autograd telemetry (`execution_order`, queue/dependency counters)
- tensor-meta invalid input boundary (`fail_closed` reason taxonomy)
- reliability gate output (`budget_id`, failing scenario IDs, remediation hint)

## 5) Conformance + Benchmark Integration Hooks

Conformance hooks:
- `run_scalar_conformance`
- `run_tensor_meta_conformance`
- `run_dispatch_conformance`
- `emit_e2e_forensics_matrix_filtered`
- `triage_forensics_failures`
- `build_failure_forensics_index`
- `check_reliability_budgets`

Benchmark hooks (packet-level):
- scalar microbench path in `ft-conformance` (`microbench_ns` metrics)
- reliability report references method-stack/perf artifacts for gate correlation

## 6) N/A Cross-Cutting Validation Note

This implementation-plan artifact is docs/planning only for subtask D.
Execution evidence is deferred to:
- `bd-3v0.12.5` (unit/property with detailed logs)
- `bd-3v0.12.6` (differential/metamorphic/adversarial)
- `bd-3v0.12.7` (e2e scenarios + replay/forensics logs)
