# FT-P2C-001 — Security + Compatibility Threat Model

Packet scope: Tensor metadata + storage core  
Subtask: `bd-3v0.12.3`

## 1) Threat Model Scope

Protected invariants:
- metadata shape/stride/index safety
- deterministic scalar/autograd behavior under strict/hardened modes
- fail-closed compatibility checks for dtype/device mismatch
- deterministic replay evidence for any policy override or recovery path

Primary attack surfaces:
- malformed tensor metadata payloads (rank/stride/offset/index overflow)
- dispatch mode split misuse (strict vs hardened fallback drift)
- replay/log tampering via missing forensic fields
- compatibility mismatch branches (dtype/device) under-specified in current fixture corpus

## 2) Compatibility Envelope and Mode-Split Fail-Closed Rules

| Boundary | Strict mode | Hardened mode | Fail-closed rule |
|---|---|---|---|
| metadata validation (`TensorMeta`) | reject invalid rank/stride/overflow inputs | same behavior | invalid metadata never auto-repaired |
| scalar/autograd core math semantics | deterministic output + gradients | same math semantics | no numeric behavior drift permitted |
| dispatch fallback branch | reject incompatible composite routing | bounded fallback only where explicitly encoded | unknown/incompatible keysets always fail |
| dtype/device compatibility | reject mismatch | reject mismatch | no silent coercion in this packet |
| replay log envelope fields | required | required | missing replay fields treated as reliability gate violation |

## 3) Packet-Specific Abuse Classes and Defensive Controls

| Threat ID | Abuse class | Attack surface | Impact if unmitigated | Defensive control | Strict/Hardened expectation | Unit/property fixture mapping | Failure-injection e2e scenario seed(s) |
|---|---|---|---|---|---|---|---|
| `THR-001` | rank/stride mismatch injection | metadata constructor / validation | invalid metadata enters kernel path | `RankStrideMismatch` fail-closed gate | strict=fail, hardened=fail | `ft_core::index_rank_and_bounds_are_guarded` + invalid rank fixture | `tensor_meta/strict:invalid_rank_stride_mismatch`=`9353830229903822145`, `tensor_meta/hardened:invalid_rank_stride_mismatch`=`5997540812546318856` |
| `THR-002` | offset/index overflow pressure | linear index arithmetic path | OOB or wrapped index | `StrideOverflow` / `StorageOffsetOverflow` fail-closed gate | strict=fail, hardened=fail | `ft_core::custom_strides_validate_and_index_into_storage` + invalid overflow fixture | `tensor_meta/strict:invalid_storage_offset_overflow`=`11931105988727078667`, `tensor_meta/hardened:invalid_storage_offset_overflow`=`1156477838142738040` |
| `THR-003` | mode-split dispatch abuse | composite route fallback branch | strict drift hidden by fallback | strict hard-fail; hardened bounded fallback + explicit telemetry | strict=fail, hardened=fallback allowed by policy | `ft_dispatch::strict_mode_rejects_composite_fallback`, `ft_dispatch::hardened_mode_allows_composite_fallback` | `dispatch_key/strict:composite_route_mode_split` (seed in e2e matrix), `dispatch_key/hardened:composite_route_mode_split` |
| `THR-004` | replay evidence omission/tampering | structured forensic logs | unreplayable failures and audit loss | reliability gate enforces required fields and reason taxonomy | same in both modes | `check_reliability_budgets` + forensics index tests | `e2e_matrix_full_v1.jsonl` full-window gate run |
| `THR-005` | dtype/device mismatch bypass | compatibility boundary (`ensure_compatible`, device guard) | semantic mismatch may propagate | explicit fail-closed mismatch errors | strict=fail, hardened=fail | candidate negative fixtures pending | candidate seeds: `tensor_meta/strict:compat_device_mismatch_candidate`=`1609459201001`, `tensor_meta/hardened:compat_device_mismatch_candidate`=`1609459201002` |

## 4) Mandatory Forensic Logging + Replay Artifacts for Incidents

For every security/compat incident, logs must include:
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `reason_code`
- `artifact_refs`
- `replay_command`
- `env_fingerprint`

Required artifact linkage chain:
1. e2e log entry (`artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl`)
2. crash triage summary (`artifacts/phase2c/e2e_forensics/crash_triage_full_v1.json`)
3. failure index envelope (`artifacts/phase2c/e2e_forensics/failure_forensics_index_v1.json`)
4. reliability budget report (`artifacts/phase2c/e2e_forensics/reliability_gate_report_v1.json`)

## 5) Residual Risks and Deferred Controls

Residual risks:
- device/dtype mismatch adversarial scenarios are still represented as candidates, not active fixtures.
- symbolic-shape parity and large-shape oracle timeout envelopes are deferred outside this packet.

Deferred controls and ownership:
- close `THR-005` fixture gap via `bd-3v0.23.10` follow-on and packet closeout `bd-3v0.12.9`.
- integrate packet-level timeout/cancel abuse taxonomy under `bd-3v0.21` follow-on.
- continue RaptorQ durability-evidence coupling under `bd-3v0.9` and packet final evidence bead `bd-3v0.12.9`.

## 6) Cross-Cutting Validation Update (2026-02-14)

Execution evidence link status:
- `bd-3v0.12.5` complete:
  - `artifacts/phase2c/FT-P2C-001/unit_property_quality_report_v1.json`
  - `artifacts/phase2c/FT-P2C-001/unit_property_coverage_v1.json`
- `bd-3v0.12.6` complete:
  - `artifacts/phase2c/FT-P2C-001/differential_packet_report_v1.json`
  - `artifacts/phase2c/FT-P2C-001/differential_reconciliation_v1.md`
- `bd-3v0.12.7` complete:
  - `artifacts/phase2c/e2e_forensics/e2e_matrix_ft_p2c_001.jsonl`
  - `artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_001_v1.json` (`packet_filter=FT-P2C-001`)
  - `artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_001_v1.json`
  - `artifacts/phase2c/FT-P2C-001/e2e_replay_forensics_linkage_v1.json`
  - `artifacts/phase2c/FT-P2C-001/e2e_replay_forensics_linkage_v1.md`

Threat-control status:
- `THR-001`, `THR-002`, and `THR-004` are covered by active unit/property + differential + forensics artifacts.
- `THR-005` remains explicitly deferred under `GAP-UX-001` until mismatch fixtures are added to packet differential/e2e suites.
