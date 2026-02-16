# FT-P2C-007 — Security + Compatibility Threat Model

Packet: Device guard and backend transitions  
Scope owner bead: `bd-3v0.18.3`

## Boundary and Assets

In-scope asset surfaces:
- device-guard validation and same-device checks (`DeviceGuard`, `ensure_same_device`)
- dispatch keyset validation (`EmptySet`, `NoTypeKey`, `NoBackendKey`, unknown-key rejection)
- backend/type resolution and mode split (`strict` vs `hardened`) for composite/autograd paths
- deterministic dispatch decision evidence (`selected_key`, `backend_key`, `fallback_used`, `kernel`)
- packet artifacts under `artifacts/phase2c/FT-P2C-007/`

Out-of-scope (explicit, non-silent):
- non-CPU backend execution expansion (CUDA/XPU/MPS/etc.)
- heterogeneous stream/device-guard implementations outside current CPU packet
- broad backend/domain closure beyond explicit packet-007 deferred controls

## Compatibility Envelope and Mode-Split Fail-Closed Rules

| Boundary | Strict mode | Hardened mode | Fail-closed rule |
|---|---|---|---|
| device guard checks | exact device match required | exact device match required | mismatch is terminal (`DeviceError::Mismatch`) |
| autograd/backend coupling | `AutogradCPU` requires `CPU` backend key | same requirement | no backend synthesis/repair |
| composite/backend transition | composite fallback is forbidden | bounded fallback allowed only with backend presence + evidence | unknown/incompatible keysets remain terminal |
| malformed keysets | empty/no-type/no-backend/unknown keyset fails closed | same | never coerce malformed keyset into executable route |
| dtype/device compatibility | mismatch is terminal | mismatch is terminal | no implicit cast or device migration |
| unsupported backend domain | explicit non-support | explicit non-support | unknown unsupported backend families fail closed |

## Threat Classes and Policy Response

| Threat ID | Abuse Class | Entry Vector | Strict Response | Hardened Response | Evidence/Test Hooks | Deterministic Scenario Seed(s) |
|---|---|---|---|---|---|---|
| `T007-01` | device-guard bypass attempt | mismatched tensor/guard device pairing | fail closed (`DeviceError::Mismatch`) | same fail-closed behavior | `ft_device::ensure_same_device`; conformance case `device_mismatch_fail_closed` | `dispatch_key/strict:device_mismatch_fail_closed`=`6866107161847777644`, `dispatch_key/hardened:device_mismatch_fail_closed`=`12535334678939657837` |
| `T007-02` | composite fallback escalation | keyset includes `Composite*` + backend-select path | reject composite/backend fallback | allow bounded fallback only under allowlist constraints and explicit `fallback_used` evidence | `ft_dispatch::strict_mode_rejects_composite_fallback`, `ft_dispatch::hardened_mode_allows_composite_fallback`, `ft_dispatch::prop_mode_split_for_composite_keysets` | `dispatch_key/strict:composite_route_mode_split`=`14228129716249401336`, `dispatch_key/hardened:composite_route_mode_split`=`2146157517907283417` |
| `T007-03` | autograd/backend incompatibility | `AutogradCPU` provided without `CPU` backend key | fail closed (`IncompatibleSet`) | same fail-closed behavior | `ft_dispatch::validate_requires_cpu_for_autograd`, `ft_dispatch::prop_validate_requires_cpu_for_autograd` | `dispatch_key/strict:autograd_without_cpu_fail_closed`=`7393412218162034649`, `dispatch_key/hardened:autograd_without_cpu_fail_closed`=`8135661821154981850` |
| `T007-04` | keyset poisoning / parser-state abuse | empty, no-type, no-backend, unknown dispatch key | fail closed with deterministic error taxonomy | same fail-closed behavior | `ft_dispatch::unknown_bits_fail_closed`; conformance cases `empty_keyset_fail_closed`, `no_type_key_fail_closed`, `no_backend_key_fail_closed`, `unknown_dispatch_key_fail_closed` | `dispatch_key/strict:empty_keyset_fail_closed`=`17570381862974015464`, `dispatch_key/hardened:empty_keyset_fail_closed`=`14701668349512464216`, `dispatch_key/strict:unknown_dispatch_key_fail_closed`=`16228262575860461841`, `dispatch_key/hardened:unknown_dispatch_key_fail_closed`=`6481405883647676142` |
| `T007-05` | silent dtype coercion | binary dispatch request with incompatible dtype pair | fail closed before kernel | same fail-closed behavior | conformance case `dtype_mismatch_fail_closed`; unit/property expansion in `bd-3v0.18.5` | `dispatch_key/strict:dtype_mismatch_fail_closed`=`1945855344420388393`, `dispatch_key/hardened:dtype_mismatch_fail_closed`=`18297654663329102904` |
| `T007-06` | backend-domain confusion | unsupported non-CPU backend key families presented in packet-007 scope | explicit non-support + fail closed | explicit non-support + fail closed | deferred expansion controls in `bd-3v0.18.6`/`.7`; contract gap marker `GAP-DISPATCH-007-BACKEND-DOMAIN` | `dispatch_key/strict:non_cpu_backend_key_candidate`=`201313005`, `dispatch_key/hardened:non_cpu_backend_key_candidate`=`201313006` |

## Adversarial Fixture and Failure-Injection E2E Plan

Current implemented adversarial anchors:
- dispatch mode split: strict reject / hardened bounded fallback (`composite_route_mode_split`)
- autograd-without-backend fail-closed (`autograd_without_cpu_fail_closed`)
- malformed keysets fail-closed (`empty_keyset_fail_closed`, `unknown_dispatch_key_fail_closed`)
- dtype/device mismatch fail-closed (`dtype_mismatch_fail_closed`, `device_mismatch_fail_closed`)

Execution-bead ownership for closure evidence:
- unit/property + deterministic structured logs: `bd-3v0.18.5`
- differential/metamorphic/adversarial packet checks: `bd-3v0.18.6`
- e2e failure-injection and replay/forensics artifacts: `bd-3v0.18.7`

## Mandatory Forensic Logging and Replay Artifacts

Required per-incident fields:
- `suite_id`
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `reason_code`
- `artifact_refs`
- `replay_command`
- `env_fingerprint`
- `outcome`

Packet-007-specific forensic additions:
- `dispatch_key`
- `backend_key`
- `selected_kernel`
- `keyset_bits`
- `fallback_path`
- `device_pair`
- `dtype_pair`
- `error_message`
- `contract_ids`

Artifact chain requirements:
1. packet threat model (`artifacts/phase2c/FT-P2C-007/threat_model.md`)
2. packet contract ledger (`artifacts/phase2c/FT-P2C-007/contract_table.md`)
3. packet e2e forensics JSONL slice (owned by `bd-3v0.18.7`)
4. packet differential report/reconciliation artifacts (owned by `bd-3v0.18.6`)
5. packet final evidence and RaptorQ closure artifacts (owned by `bd-3v0.18.9`)

## Release-Gate Implications

1. Any non-allowlisted hardened recovery behavior is release-blocking.
2. Unknown/incompatible dispatch/backend/device feature paths must fail closed in both modes.
3. No packet-007 closure claim is valid without deterministic adversarial and e2e forensics evidence.
4. Deferred backend-domain expansion must stay explicit; silent acceptance is forbidden.

## N/A Cross-Cutting Validation Note

This bead output is docs/planning for packet subtask C (`bd-3v0.18.3`).
Execution evidence ownership is explicitly delegated to:
- unit/property: `bd-3v0.18.5`
- differential/metamorphic/adversarial: `bd-3v0.18.6`
- e2e/replay forensics: `bd-3v0.18.7`
- optimization/isomorphism and final packet closure: `bd-3v0.18.8`, `bd-3v0.18.9`
