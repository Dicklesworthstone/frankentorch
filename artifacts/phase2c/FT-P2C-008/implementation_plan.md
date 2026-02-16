# FT-P2C-008 — Rust Implementation Plan + Module Boundary Skeleton

Packet: NN module/state contract first-wave  
Subtask: `bd-3v0.19.4`

## 1) Module Boundary Justification

| Crate/module | Ownership | Why this boundary is required | Integration seam |
|---|---|---|---|
| `ft-nn` (planned) module state core (`ModuleNode`, parameter/buffer registries) | packet-local module/state semantics | isolates module namespace, registration, traversal, and mode propagation invariants from serialization/dispatch plumbing | consumed by packet fixtures and top-level API facade |
| `ft-nn` (planned) state export/load layer (`state_dict`, `load_state_dict`) | strict/hardened compatibility split for state payloads | centralizes key-shape compatibility checks, mismatch taxonomy, and hook envelopes | invoked by API/session-level module interfaces and conformance harness |
| `ft-nn` (planned) compatibility helpers (`consume_prefix_*`, hook trace boundary) | deterministic adaptation envelope for wrapped checkpoints | prevents ad-hoc compatibility rewrites at call sites and keeps hardened behavior auditable | consumed by load pipeline before strictness evaluation |
| `ft-serialize` | durability and checkpoint evidence envelope | packet state payloads must preserve deterministic serialization and sidecar/proof requirements inherited from packet-006 | consumed by packet final evidence pack and durability gates |
| `ft-api` facade | session-facing module/state entry points | keeps user-observable entry contracts centralized and mode-aware (`strict`/`hardened`) | delegates to `ft-nn` state core and emits runtime/evidence logs |
| `ft-conformance` fixture family (`FT-P2C-008`) | packet strict/hardened parity and adversarial gate execution | single source for packet state behavior assertions and deterministic replay envelope | emits differential slices, e2e forensics, reliability and closure artifacts |

## 2) Low-Risk Implementation Sequence (One Optimization Lever per Step)

| Step | Change scope | Semantic risk strategy | Single optimization lever |
|---|---|---|---|
| `S1` | lock registration and module traversal contracts | prove deterministic namespace/order invariants before enabling load/transfer paths | memoized module-path expansion for repeated traversal checks |
| `S2` | implement state export/load strict path (`state_dict`, `load_state_dict(strict=true)`) | require deterministic mismatch taxonomy before adding hardened branches | precomputed state key-index map for strict compatibility checks |
| `S3` | add hardened bounded compatibility pathways (`strict=false`, prefix normalization, hook trace envelope) | allowlist hardened-only behaviors and enforce deterministic telemetry before broader testing | normalized prefix cache for repetitive wrapped-state loads |
| `S4` | wire unit/property + differential/metamorphic/adversarial packet checks | block non-allowlisted hardened drift prior to e2e closure | packet-filtered differential comparator execution to reduce noise |
| `S5` | emit packet-scoped e2e replay/forensics and closure-ready artifacts | require one-command replay + triage/index outputs before final evidence bead | packet-filtered JSONL emission and indexed scenario lookup |

## 3) Detailed Test Implementation Plan

### 3.1 Unit/property suite plan

- `ft-nn` (planned packet implementation surface)
  - `register_parameter_rejects_invalid_name`
  - `register_buffer_tracks_persistence_flag`
  - `state_dict_includes_parameters_and_persistent_buffers`
  - `state_dict_nested_prefixes_are_stable`
  - `named_modules_prefix_order_is_stable`
  - `train_propagates_recursively`
  - `eval_is_train_false_alias`
  - `load_state_dict_strict_rejects_unexpected_keys`
  - `load_state_dict_strict_rejects_missing_keys`
  - `load_state_dict_rejects_shape_mismatch`
  - `prefix_consumption_maps_ddp_state_dict_keys`
  - `state_dict_hooks_fire_in_order`
  - `load_state_dict_pre_post_hooks_emit_trace`
  - `module_to_preserves_keyset_and_shapes`
  - `load_state_dict_assign_rejects_incompatible_tensor_shape`
- `ft-conformance` packet suite additions
  - strict/hardened packet conformance checks for all `FTP2C008-B01..B09` families

### 3.2 Differential/metamorphic/adversarial hooks

- differential source artifact:
  - `artifacts/phase2c/conformance/differential_report_v1.json`
- packet-slice artifacts (target in `bd-3v0.19.6`):
  - `artifacts/phase2c/FT-P2C-008/differential_packet_report_v1.json`
  - `artifacts/phase2c/FT-P2C-008/differential_reconciliation_v1.md`
- adversarial targets:
  - missing/unexpected key strictness bypass attempts
  - prefix spoofing and malformed normalization payloads
  - hook-driven incompatibility suppression attempts
  - transfer/cast unsupported path abuse
  - non-dict payload and incompatible assign-path abuse
- deterministic threat seeds and abuse mapping are anchored in:
  - `artifacts/phase2c/FT-P2C-008/threat_model.md`

### 3.3 E2E script plan

- packet-scoped e2e command (target in `bd-3v0.19.7`):
  - `rch exec -- env CARGO_TARGET_DIR=target_amberpelican_008 cargo run -p ft-conformance --bin run_e2e_matrix -- --mode both --packet FT-P2C-008 --output artifacts/phase2c/e2e_forensics/ft-p2c-008.jsonl --print-full-log`
- packet triage/index commands (target in `bd-3v0.19.7`):
  - `rch exec -- env CARGO_TARGET_DIR=target_amberpelican_008 cargo run -p ft-conformance --bin triage_forensics_failures -- --input artifacts/phase2c/e2e_forensics/ft-p2c-008.jsonl --output artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_008_v1.json --packet FT-P2C-008`
  - `rch exec -- env CARGO_TARGET_DIR=target_amberpelican_008 cargo run -p ft-conformance --bin build_failure_forensics_index -- --e2e artifacts/phase2c/e2e_forensics/ft-p2c-008.jsonl --triage artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_008_v1.json --output artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_008_v1.json`

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

NN module/state additions:
- `module_path`
- `state_key`
- `state_key_kind`
- `strict_flag`
- `assign_flag`
- `missing_keys`
- `unexpected_keys`
- `incompatible_shapes`
- `hook_trace`
- `prefix_normalization_applied`
- `training_flag_transition`

## 5) Conformance + Benchmark Integration Hooks

Conformance hooks:
- strict/hardened packet conformance suites for `FT-P2C-008`
- packet differential report + reconciliation artifacts
- packet e2e replay + triage + failure-index outputs
- reliability budget policy checks (`check_reliability_budgets`)
- packet/global validator integration (`validate_phase2c_artifacts`) at packet closure stage

Benchmark hooks:
- module traversal overhead across nested module trees (p50/p95/p99)
- strict load mismatch classification overhead under adversarial keysets
- hardened prefix/hook pathway overhead with deterministic replay metadata
- transfer/cast pathway overhead under representative module-state workloads

## 6) N/A Cross-Cutting Validation Note

This implementation-plan artifact is docs/planning for subtask D (`bd-3v0.19.4`).
Execution evidence ownership is explicitly delegated to:
- `bd-3v0.19.5` (unit/property + structured logs)
- `bd-3v0.19.6` (differential/metamorphic/adversarial)
- `bd-3v0.19.7` (e2e replay/forensics logging)
- `bd-3v0.19.8` (optimization/isomorphism)
- `bd-3v0.19.9` (final evidence pack)
