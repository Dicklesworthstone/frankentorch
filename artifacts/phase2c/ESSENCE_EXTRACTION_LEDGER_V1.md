# ESSENCE_EXTRACTION_LEDGER_V1.md

Date: 2026-02-14
Schema: `ft-essence-ledger-v1`
Scope: Phase-2C packet extraction for strict/hardened clean-room parity work.

Purpose: lock behaviorally critical legacy semantics into a traceable ledger so implementation, conformance, and e2e work can proceed without re-opening legacy code for every decision.

## 1. Ledger Fields (Normative)

Each row MUST include:
1. `ledger_id`
2. `packet_id`
3. `legacy_anchor`
4. `observable_contract`
5. `strict_expectation`
6. `hardened_expectation`
7. `uncertainty_tag`
8. `unit_property_targets`
9. `differential_targets`
10. `e2e_seed`
11. `structured_log_contract`
12. `evidence_status`

`uncertainty_tag` vocabulary:
- `low`: anchor and observable behavior are stable and directly evidenced.
- `medium`: behavior is clear but full global parity surface is not yet extracted.
- `high`: expected behavior inferred from partial anchors and requires follow-up extraction.

## 2. Extraction Ledger

| ledger_id | packet_id | legacy_anchor | observable_contract | strict_expectation | hardened_expectation | uncertainty_tag | unit_property_targets | differential_targets | e2e_seed | structured_log_contract | evidence_status |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `L-001-META-RANK` | `FT-P2C-001` | `c10/core/TensorImpl.h` rank/stride metadata regions | shape rank must equal stride rank | fail-closed on mismatch | fail-closed on mismatch | low | `ft_core::TensorMeta::validate` rank mismatch tests | `scalar_dac` scenario IDs must include packet mapping | `ft-p2c-001-meta-rank-v1` | `scenario_id, seed, mode, reason_code` | implemented |
| `L-002-META-OFFSET` | `FT-P2C-001` | `storage_offset` + `sizes_and_strides_` fields | storage offset arithmetic must be overflow-safe | overflow rejected | overflow rejected | low | storage offset overflow + index mapping tests | differential checks for metadata-derived scalar route correctness | `ft-p2c-001-meta-offset-v1` | `artifact_refs include FT-P2C-001 parity artifacts` | implemented |
| `L-003-VERSION-COUNTER` | `FT-P2C-001` | `VariableVersion` anchor | version behavior must be deterministic and observable | out-of-place derivation increments derived version, no silent mutation | same observable rule | medium | out-of-place version bump tests + in-place mutation tests | parity checks ensure no output/grad regression under version updates | `ft-p2c-001-version-counter-v1` | `scenario_id ties operation path + mode` | implemented |
| `L-004-STORAGE-IDENTITY` | `FT-P2C-001` | `Storage storage_` field | storage identity must be explicit for alias analysis | alias view preserves storage identity | alias view preserves storage identity | medium | `storage_id` identity + alias tests | drift checks require no value/grad mismatch under alias-preserving paths | `ft-p2c-001-storage-identity-v1` | `reason_code distinguishes alias vs out_of_place` | implemented |
| `L-005-DISPATCH-PRECEDENCE` | `FT-P2C-002` | `DispatchKey.h`, `DispatchKeySet.h`, `Dispatch.h` | key precedence must route deterministic kernel/backend path | unknown/incompatible keysets fail closed | bounded fallback only if allowlisted | low | dispatch route + unknown bits tests | differential dispatch output checks and policy checks | `ft-p2c-002-dispatch-precedence-v1` | `packet_id + comparator + drift_id` | implemented |
| `L-006-DISPATCH-MODE-SPLIT` | `FT-P2C-002` | composite/backend-select fallback anchor set | strict/hardened policy split is explicit and auditable | strict rejects composite fallback path | hardened allows bounded fallback w/ evidence | low | strict reject + hardened allow tests | `dispatch.composite_backend_fallback` allowlist classification | `ft-p2c-002-mode-split-v1` | `allowlisted flag + drift_id mandatory` | implemented |
| `L-007-AUTOGRAD-ORDER` | `FT-P2C-004` | `NodeTask`, `ReadyQueue`, `Engine::execute` | backward scheduler order and deps are deterministic | deterministic order invariant enforced | deterministic order invariant enforced | medium | scheduler dependency + order tests | differential grad/output checks for scheduler cases | `ft-p2c-004-order-v1` | `execution_order evidence in telemetry` | implemented |
| `L-008-REENTRANT-GUARD` | `FT-P2C-004` | reentrant policy paths in autograd engine | reentrant overflow policy must be mode-split and observable | strict fails on overflow | hardened applies bounded fallback + telemetry | low | strict overflow error + hardened guard-trigger tests | `autograd.reentrant_depth_bounded_fallback` allowlist classification | `ft-p2c-004-reentrant-v1` | `reason_code requires policy-specific code` | implemented |
| `L-009-SERIALIZATION-STRICT` | `FT-P2C-006` | `serialization.cpp` raw read/write anchors | incompatible payload/version must fail closed | strict rejects malformed/unknown fields | hardened still rejects incompatibility with bounded diagnostics | medium | strict unknown field/version tests | serialization suite differential invariants via deterministic hash checks | `ft-p2c-006-serialization-strict-v1` | `artifact_refs include decode proof` | implemented |
| `L-010-RAPTORQ-DURABILITY` | `FT-P2C-006` | durability sidecar policy (project doctrine) | durable artifacts require sidecar + decode proof | required | required | low | sidecar generation + deterministic proof hash tests | conformance report durability checks in packet artifacts | `ft-p2c-006-raptorq-v1` | `artifact hash + decode proof hash required` | implemented |
| `L-011-SYMBOLIC-SHAPE-GAP` | `FT-P2C-003` | `sym_*` anchors in `TensorImpl` | symbolic shape parity coverage pending full extraction | no silent acceptance of unknown symbolic behavior | same with bounded diagnostics where allowed | high | placeholder assertions plus explicit gap markers | differential checks to remain blocked until schema extraction closes | `ft-p2c-003-symbolic-gap-v1` | `reason_code must indicate extraction gap` | deferred-with-gate |
| `L-012-NN-STATE-SURFACE` | `FT-P2C-008` | `torch/nn/*` state contracts | module/state parity requires full state-dict semantics extraction | fail on unknown incompatible state | bounded hardened diagnostics without semantic drift | high | planned module-state tests | planned state-dict differential suite | `ft-p2c-008-state-v1` | `scenario_id namespace reserved` | deferred-with-gate |

## 3. Traceability Map

- Unit/property roots:
  - `crates/ft-core/src/lib.rs`
  - `crates/ft-dispatch/src/lib.rs`
  - `crates/ft-autograd/src/lib.rs`
  - `crates/ft-serialize/src/lib.rs`
- Differential roots:
  - `crates/ft-conformance/src/lib.rs`
  - `crates/ft-conformance/src/bin/run_differential_report.rs`
  - `artifacts/phase2c/conformance/differential_report_v1.json`
- E2E/log roots:
  - `crates/ft-conformance/src/bin/run_e2e_matrix.rs`
  - `artifacts/phase2c/e2e_forensics/*.jsonl` (when generated)

## 4. N/A Evidence Note (for docs-only extraction rows)

Rows marked `deferred-with-gate` are not implementation omissions.
They are sequencing placeholders with explicit parity-closure dependencies:
- `FT-P2C-003` schema ingestion closure beads.
- `FT-P2C-008` nn/state closure beads.

Until closure beads land, release gates treat those rows as open parity obligations.

## 5. Closure Rule

`bd-3v0.1` is closure-eligible only when:
- no row is missing mandatory fields,
- each non-deferred row links to concrete unit/property + differential + e2e/log targets,
- each deferred row has explicit parity-closure gating dependencies.
