# FT-P2C-002 — Rust Implementation Plan + Module Boundary Skeleton

Packet: Dispatch key model  
Subtask: `bd-3v0.13.4`

## 1) Module Boundary Justification

| Crate/module | Ownership | Why this boundary is required | Integration seam |
|---|---|---|---|
| `ft-dispatch::DispatchKey/DispatchKeySet` | key domain, bitset parsing, type/backend priority resolution | centralizes dispatch-key semantics and fail-closed policy | consumed by `ft-autograd`, `ft-conformance`, and public session APIs |
| `ft-dispatch::dispatch_scalar_binary[_with_keyset]` | mode-split route execution and decision telemetry | keeps strict/hardened routing policy explicit and auditable | calls `ft-kernel-cpu` and returns `DispatchOutcome` |
| `ft-kernel-cpu` | scalar kernel math (`add_scalar`, `mul_scalar`) | isolates arithmetic from policy/routing decisions | invoked only through dispatch layer |
| `ft-autograd::Tape` | requires-grad propagation into dispatch key selection | preserves DAC semantics while reusing dispatch routing | calls `dispatch_scalar_binary(... requires_grad)` |
| `ft-api::FrankenTorchSession` | user-facing op calls + evidence logging | ensures session-level evidence records dispatch decisions | records decision metadata in runtime ledger |
| `ft-conformance` | dispatch fixtures, differential comparators, e2e forensics | single point for parity evidence and replay contracts | runs `run_dispatch_conformance` and emits packet artifacts |

## 2) Low-Risk Implementation Sequence (One Optimization Lever per Step)

| Step | Change scope | Semantic risk strategy | Single optimization lever |
|---|---|---|---|
| `S1` | lock keyset parsing + priority invariants in `ft-dispatch` | validate fail-closed unknown-bit and empty-set behavior before route changes | compute/retain known-bit mask once per validation path |
| `S2` | finalize strict/hardened mode-split route behavior | keep strict hard-fail as canonical; gate hardened fallback behind explicit branch | avoid duplicate fallback checks in already-executable routes |
| `S3` | align autograd integration (`requires_grad` -> `AutogradCPU`) | preserve gradient semantics while adding route evidence | reuse precomputed keyset across dispatch decision + telemetry |
| `S4` | strengthen conformance/differential dispatch comparators | classify strict/hardened drifts before adding new fixtures | perform one canonical expectation parse per case |
| `S5` | expand packet e2e/replay forensics linkage | enforce deterministic scenario seeds and replay commands | append JSONL log events in a single pass per suite |

## 3) Detailed Test Implementation Plan

### 3.1 Unit/property suite plan

- `ft-dispatch`
  - `dispatch_keyset_set_algebra_is_stable`
  - `priority_resolution_prefers_autograd_cpu`
  - `backend_priority_returns_cpu`
  - `unknown_bits_fail_closed`
  - `strict_mode_rejects_composite_fallback`
  - `hardened_mode_allows_composite_fallback`
  - `dispatch_returns_kernel_metadata`
- `ft-autograd`
  - dispatch path exercised through tape binary ops with `requires_grad=true`
- `ft-api`
  - session evidence entries include dispatch decision metadata

### 3.2 Differential/metamorphic/adversarial hooks

- differential packet evidence source:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
- dispatch adversarial candidates (to be promoted under `bd-3v0.13.6`):
  - `dispatch_key/*:unknown_bits_mask_candidate`
  - `dispatch_key/*:incompatible_autograd_without_cpu_candidate`
- hardened allowlist drift anchor:
  - `dispatch.composite_backend_fallback`

### 3.3 E2E script plan

- packet-scoped e2e run command:
  - `cargo run -p ft-conformance --bin run_e2e_matrix -- --mode both --packet FT-P2C-002 --output artifacts/phase2c/e2e_forensics/ft-p2c-002.jsonl`
- replay commands are sourced from scenario log entries and triaged via failure-index tooling.

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

Dispatch-specific instrumentation points:
- resolved `selected_key`
- resolved `backend_key`
- `keyset_bits`
- `fallback_used`
- mode-split reason (`strict_fail` vs `hardened_fallback`)

## 5) Conformance + Benchmark Integration Hooks

Conformance hooks:
- `run_dispatch_conformance`
- differential comparator branches in `run_differential_conformance`
- `emit_e2e_forensics_matrix_filtered`
- `triage_forensics_failures`
- `build_failure_forensics_index`

Benchmark hooks:
- scalar microbench path (`microbench_ns`) as coarse dispatch-route sentinel
- packet-level fallback-rate counters in dispatch forensic logs (strict should remain zero)

## 6) N/A Cross-Cutting Validation Note

This implementation-plan artifact is docs/planning only for subtask D.
Execution evidence is deferred to:
- `bd-3v0.13.5` (unit/property with detailed logs)
- `bd-3v0.13.6` (differential/metamorphic/adversarial)
- `bd-3v0.13.7` (e2e scenarios + replay/forensics logs)
