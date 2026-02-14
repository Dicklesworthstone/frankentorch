# FT-P2C-002 — Security + Compatibility Threat Model

Packet scope: Dispatch key model  
Subtask: `bd-3v0.13.3`

## 1) Threat Model Scope

Protected invariants:
- deterministic dispatch-key precedence and backend projection
- strict/hardened mode split with explicit fail-closed boundaries
- fail-closed handling for unknown/incompatible keysets
- deterministic forensic replay envelope for dispatch incidents

Primary attack surfaces:
- raw-bitset keyset corruption (`UnknownBits`) and compatibility misuse
- composite/backend-select downgrade drift across strict/hardened policy boundary
- key-priority ordering drift relative to legacy PyTorch behavior
- missing forensic fields preventing replayable diagnosis

## 2) Compatibility Envelope and Mode-Split Fail-Closed Rules

| Boundary | Strict mode | Hardened mode | Fail-closed rule |
|---|---|---|---|
| keyset parsing (`from_bits_checked`) | reject unknown bits | reject unknown bits | no unknown bit auto-repair |
| type-key precedence | deterministic fixed ordering | same ordering | empty/non-type keysets fail |
| backend projection | executable backend required | executable backend required | no backend key -> terminal error |
| composite/backend-select routing | reject fallback path | bounded fallback allowed | unknown/incompatible keysets always fail |
| replay log envelope fields | required | required | missing fields are reliability-gate violations |

## 3) Packet-Specific Abuse Classes and Defensive Controls

| Threat ID | Abuse class | Attack surface | Impact if unmitigated | Defensive control | Strict/Hardened expectation | Unit/property fixture mapping | Failure-injection e2e scenario seed(s) |
|---|---|---|---|---|---|---|---|
| `THR-201` | unknown key bit injection | `DispatchKeySet::from_bits_checked` | non-deterministic route or silent drift | `UnknownBits` fail-closed gate | strict=fail, hardened=fail | `ft_dispatch::unknown_bits_fail_closed` | candidates: `dispatch_key/strict:unknown_bits_mask_candidate`=`201313001`, `dispatch_key/hardened:unknown_bits_mask_candidate`=`201313002` |
| `THR-202` | precedence drift | `highest_priority_type_id` ordering | wrong kernel route chosen | explicit priority table + conformance parity checks | strict=deterministic, hardened=deterministic | `ft_dispatch::priority_resolution_prefers_autograd_cpu`, dispatch conformance fixtures | `dispatch_key/strict:strict_autograd_route`=`12780237016247668875`, `dispatch_key/hardened:strict_autograd_route`=`456065680046437289` |
| `THR-203` | strict-mode downgrade via fallback | composite/backend-select route selection | strict compatibility drift hidden by fallback | mode-split policy: strict hard-fail, hardened bounded fallback | strict=error, hardened=fallback allowed with evidence | `ft_dispatch::strict_mode_rejects_composite_fallback`, `ft_dispatch::hardened_mode_allows_composite_fallback` | `dispatch_key/strict:composite_route_mode_split`=`14228129716249401336`, `dispatch_key/hardened:composite_route_mode_split`=`2146157517907283417` |
| `THR-204` | incompatible autograd/backend keyset | compatibility validator | undefined route behavior | `validate_for_scalar_binary` fail-closed compatibility checks | strict=fail, hardened=fail | candidate adversarial fixture for `DispatchKeyError::IncompatibleSet` (owned by `bd-3v0.13.6`) | candidates: `dispatch_key/strict:incompatible_autograd_without_cpu_candidate`=`201313003`, `dispatch_key/hardened:incompatible_autograd_without_cpu_candidate`=`201313004` |
| `THR-205` | replay evidence omission/tampering | structured dispatch forensic logs | unreplayable failures / audit gaps | required deterministic log contract + reliability gates | same in both modes | dispatch conformance forensic log contract (`StructuredCaseLog`) | full-suite e2e log gates in `e2e_matrix_full_v1.jsonl` |

## 4) Mandatory Forensic Logging + Replay Artifacts for Incidents

For every dispatch security/compat incident, logs must include:
- `suite_id`
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `selected_key`
- `backend_key`
- `keyset_bits`
- `fallback_used`
- `reason_code`
- `artifact_refs`
- `replay_command`
- `env_fingerprint`

Required artifact linkage chain:
1. e2e log entry (`artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl`)
2. failure triage (`artifacts/phase2c/e2e_forensics/crash_triage_full_v1.json`)
3. failure index envelope (`artifacts/phase2c/e2e_forensics/failure_forensics_index_v1.json`)
4. reliability budget report (`artifacts/phase2c/e2e_forensics/reliability_gate_report_v1.json`)

## 5) Residual Risks and Deferred Controls

Residual risks:
- key-domain scope remains CPU + autograd CPU for this packet.
- explicit adversarial fixtures for unknown/incompatible raw keysets are documented but still candidate-level.

Deferred controls and ownership:
- complete adversarial differential coverage under `bd-3v0.13.6`.
- attach dedicated packet e2e fail-injection traces under `bd-3v0.13.7`.
- extend non-CPU dispatch key families under `FT-P2C-007`.

## 6) N/A Cross-Cutting Validation Note

This risk note is docs/planning evidence for packet subtask C.
Execution evidence remains owned by:
- `bd-3v0.13.5` (unit/property + structured logs)
- `bd-3v0.13.6` (differential/metamorphic/adversarial)
- `bd-3v0.13.7` (e2e/replay forensics)
