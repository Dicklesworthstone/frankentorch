# FT-P2C-007 — Behavior Extraction Ledger

Packet: Device guard and backend transitions  
Legacy anchor map: `artifacts/phase2c/FT-P2C-007/legacy_anchor_map.md`

## Behavior Families (Nominal, Edge, Adversarial)

| Behavior ID | Path class | Legacy anchor family | Strict expectation | Hardened expectation | Candidate unit/property assertions | E2E scenario seed(s) |
|---|---|---|---|---|---|---|
| `FTP2C007-B01` | nominal | `DeviceGuard` / `OptionalDeviceGuard` lifecycle (`c10/core/DeviceGuard.h`) | matching-device guard checks pass with no fallback mutation | same | `ft_device::guard_accepts_matching_device`, `ft_device::same_device_check_returns_cpu` | `dispatch_key/strict:strict_cpu_route`=`6654506012862553729`, `dispatch_key/hardened:strict_cpu_route`=`7720564810062027209` |
| `FTP2C007-B02` | nominal | autograd backend mapping (`AutogradCPU` -> `CPU`) and direct autograd route | autograd route resolves deterministically to `AutogradCPU` with backend `CPU` | same | `ft_dispatch::dispatch_returns_kernel_metadata`, `ft_conformance::strict_dispatch_conformance_is_green` | `dispatch_key/strict:strict_autograd_route`=`12780237016247668875`, `dispatch_key/hardened:strict_autograd_route`=`456065680046437289` |
| `FTP2C007-B03` | edge (mode split) | composite alias tiers + backend fallback (`OperatorEntry.cpp` dispatch-table precedence) | composite/backend fallback is rejected fail-closed | bounded fallback to backend key is allowed and flagged as fallback | `ft_dispatch::strict_mode_rejects_composite_fallback`, `ft_dispatch::hardened_mode_allows_composite_fallback`, `ft_dispatch::prop_mode_split_for_composite_keysets` | `dispatch_key/strict:composite_route_mode_split`=`14228129716249401336`, `dispatch_key/hardened:composite_route_mode_split`=`2146157517907283417` |
| `FTP2C007-B04` | edge (keyset compatibility) | `AutogradCPU requires CPU backend availability` invariant | keyset lacking backend bit is rejected | same fail-closed rejection | `ft_dispatch::validate_requires_cpu_for_autograd`, conformance adversarial keyset probe | `dispatch_key/strict:autograd_without_cpu_fail_closed`=`7393412218162034649`, `dispatch_key/hardened:autograd_without_cpu_fail_closed`=`8135661821154981850` |
| `FTP2C007-B05` | adversarial (malformed keysets) | dispatch-key parsing + runtime keyset validation | empty/unknown/no-backend/no-type keysets fail closed | same | conformance cases `empty_keyset_fail_closed`, `no_type_key_fail_closed`, `no_backend_key_fail_closed`, `unknown_dispatch_key_fail_closed` | `dispatch_key/strict:empty_keyset_fail_closed`=`17570381862974015464`, `dispatch_key/strict:unknown_dispatch_key_fail_closed`=`16228262575860461841`, `dispatch_key/hardened:empty_keyset_fail_closed`=`14701668349512464216`, `dispatch_key/hardened:unknown_dispatch_key_fail_closed`=`6481405883647676142` |
| `FTP2C007-B06` | adversarial (dtype compatibility) | binary-op compatibility + iterator-level fail-closed contracts | dtype mismatch is rejected before kernel execution | same | conformance case `dtype_mismatch_fail_closed` | `dispatch_key/strict:dtype_mismatch_fail_closed`=`1945855344420388393`, `dispatch_key/hardened:dtype_mismatch_fail_closed`=`18297654663329102904` |
| `FTP2C007-B07` | adversarial (device compatibility) | `check_and_update_common_device`, `common_device_check_failure`, TensorIterator same-device enforcement | cross-device pair is rejected with stable mismatch reason | same | conformance case `device_mismatch_fail_closed` | `dispatch_key/strict:device_mismatch_fail_closed`=`6866107161847777644`, `dispatch_key/hardened:device_mismatch_fail_closed`=`12535334678939657837` |
| `FTP2C007-B08` | deferred parity edge | full backend-domain expansion beyond CPU-focused packet | unsupported backend families remain explicit non-goals and must fail closed | same | deferred to downstream backend-expansion beads under FT-P2C-007/FT-P2C-008 chain | candidate seeds: `dispatch_key/strict:non_cpu_backend_key_candidate`=`201313005`, `dispatch_key/hardened:non_cpu_backend_key_candidate`=`201313006` |

## Logging Field Expectations by Behavior Family

Mandatory deterministic replay fields (all device-guard/backend-transition families):
- `suite_id`
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `outcome`
- `reason_code`

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

Anchors:
- `crates/ft-device/src/lib.rs`
- `crates/ft-dispatch/src/lib.rs`
- `crates/ft-conformance/fixtures/dispatch_key_cases.json`
- `crates/ft-conformance/src/lib.rs`
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`
- `artifacts/phase2c/e2e_forensics/ft-p2c-005.jsonl`

## N/A Cross-Cutting Validation Note

This ledger is docs/planning only for packet subtask A (`bd-3v0.18.1`).  
Execution-evidence ownership is carried by downstream packet beads:
- contract/invariant closure: `bd-3v0.18.2`
- security/compatibility threat model: `bd-3v0.18.3`
- implementation boundaries: `bd-3v0.18.4`
- unit/property + structured logging: `bd-3v0.18.5`
- differential/metamorphic/adversarial validation: `bd-3v0.18.6`
- e2e scripts + replay/forensics logging: `bd-3v0.18.7`
- optimization/isomorphism proof: `bd-3v0.18.8`
- final evidence pack + RaptorQ closure: `bd-3v0.18.9`
