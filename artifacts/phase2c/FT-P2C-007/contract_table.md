# FT-P2C-007 — Contract Table + Strict/Hardened Invariant Spec

Packet: Device guard and backend transitions  
Dependencies: `bd-3v0.18.1` behavior extraction ledger + legacy anchor map

## Machine-Checkable Contract Row Schema

Each contract row is complete only if it defines:
- preconditions
- postconditions
- invariant class ID(s)
- strict-mode semantics
- hardened-mode semantics
- fail-closed boundary decision
- unit/property mapping
- differential/metamorphic/adversarial intent
- e2e scenario ID mapping
- drift posture (`forbidden`, `allowlisted_hardened_only`, `deferred_with_gap_id`)

## Contract Rows

| Contract ID | Behavior ID | Preconditions | Postconditions | Invariant class | Strict semantics | Hardened semantics | Fail-closed boundary | Unit/property mapping | Differential/metamorphic/adversarial intent | E2E scenario IDs | Drift posture |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `DEVICE-GUARD-001` | `FTP2C007-B01` | `DeviceGuard` target device and tensor devices are explicitly set and equal | guard validation succeeds; same-device check yields deterministic `Device::Cpu` for scoped CPU packet cases | `FT-I1`, `FT-I6` | accept only exact device match | same as strict | any guard/device mismatch is terminal (`DeviceError::Mismatch`) | `ft_device::guard_accepts_matching_device`, `ft_device::same_device_check_returns_cpu` | differential checks compare local guard acceptance and dispatch-case prerequisites against packet fixture expectations; metamorphic probe replays equivalent CPU guard scopes with stable outcomes | `dispatch_key/strict:strict_cpu_route`, `dispatch_key/hardened:strict_cpu_route` | forbidden |
| `BACKEND-TRANSITION-002` | `FTP2C007-B02` | `requires_grad=true` route with keyset containing `AutogradCPU` and `CPU` | dispatch decision is deterministic (`selected_key=AutogradCPU`, `backend_key=CPU`) and kernel metadata is replay-stable | `FT-I1`, `FT-I3`, `FT-I6` | autograd route must resolve with no fallback | same as strict | missing/incompatible backend coupling terminates before kernel execution | `ft_dispatch::dispatch_returns_kernel_metadata`, `ft_conformance::strict_dispatch_conformance_is_green`, `ft_conformance::hardened_dispatch_conformance_is_green` | differential checks compare decision fields and outputs vs fixture oracle; metamorphic commutativity checks keep route metadata stable for equivalent operand swaps | `dispatch_key/strict:strict_autograd_route`, `dispatch_key/hardened:strict_autograd_route` | forbidden |
| `BACKEND-MODE-003` | `FTP2C007-B03` | dispatch keyset contains `CompositeExplicitAutograd` or `CompositeImplicitAutograd` with `CPU` backend availability | strict path emits explicit terminal error; hardened path permits bounded fallback with `fallback_used=true` and stable backend resolution | `FT-I3`, `FT-I6` | composite/backend fallback is forbidden and must fail closed | bounded fallback allowed only when backend key is present and evidence logging is complete | unknown/incompatible keysets stay terminal in both modes | `ft_dispatch::strict_mode_rejects_composite_fallback`, `ft_dispatch::hardened_mode_allows_composite_fallback`, `ft_dispatch::prop_mode_split_for_composite_keysets` | differential checks enforce strict error vs hardened success split; metamorphic checks vary composite explicit/implicit key while preserving hardened backend/output equivalence; adversarial probes deny fallback without CPU backend key | `dispatch_key/strict:composite_route_mode_split`, `dispatch_key/hardened:composite_route_mode_split` | allowlisted_hardened_only (`dispatch.composite_backend_fallback`, packet entry required before closure) |
| `BACKEND-KEYSET-004` | `FTP2C007-B04` | keyset includes `AutogradCPU` without `CPU` backend | dispatch validation rejects route with deterministic incompatibility reason and no kernel output | `FT-I5`, `FT-I6` | `IncompatibleSet` rejection is mandatory | same as strict | no repair path may synthesize backend capability | `ft_dispatch::validate_requires_cpu_for_autograd`, `ft_dispatch::prop_validate_requires_cpu_for_autograd` | adversarial keyset differential checks confirm stable fail-closed taxonomy across modes | `dispatch_key/strict:autograd_without_cpu_fail_closed`, `dispatch_key/hardened:autograd_without_cpu_fail_closed` | forbidden |
| `BACKEND-KEYSET-005` | `FTP2C007-B05` | keyset is empty, has no type key, has no backend key, or includes unknown dispatch key symbols | route validation fails closed with deterministic error taxonomy (`EmptySet`, `NoTypeKey`, `NoBackendKey`, `UnknownBits`/unknown-key rejection) | `FT-I5`, `FT-I6` | all malformed keyset families are terminal failures | same as strict | malformed keyset must never be coerced into executable routing | `ft_dispatch::unknown_bits_fail_closed`, `ft_conformance::strict_dispatch_conformance_is_green`, `ft_conformance::hardened_dispatch_conformance_is_green` | adversarial corpus mutates keyset families and verifies mode-invariant failure reasons; differential checks ensure no hardened acceptance drift | `dispatch_key/strict:empty_keyset_fail_closed`, `dispatch_key/hardened:empty_keyset_fail_closed`, `dispatch_key/strict:no_type_key_fail_closed`, `dispatch_key/hardened:no_type_key_fail_closed`, `dispatch_key/strict:no_backend_key_fail_closed`, `dispatch_key/hardened:no_backend_key_fail_closed`, `dispatch_key/strict:unknown_dispatch_key_fail_closed`, `dispatch_key/hardened:unknown_dispatch_key_fail_closed` | forbidden |
| `BACKEND-COMPAT-006` | `FTP2C007-B06` | binary op dispatch request where `lhs_dtype != rhs_dtype` under same-device CPU path | request is rejected before kernel execution with deterministic incompatibility outcome | `FT-I5`, `FT-I6` | dtype mismatch is terminal fail-closed | same as strict | no implicit dtype coercion permitted in packet scope | planned unit/property in `bd-3v0.18.5`; current anchor: conformance case `dtype_mismatch_fail_closed` | adversarial differential checks vary dtype pairs and require invariant fail-closed behavior in both modes | `dispatch_key/strict:dtype_mismatch_fail_closed`, `dispatch_key/hardened:dtype_mismatch_fail_closed` | forbidden |
| `DEVICE-COMPAT-007` | `FTP2C007-B07` | binary op dispatch request where `lhs_device != rhs_device` | request is rejected with deterministic device-mismatch reason; no kernel selected | `FT-I5`, `FT-I6` | cross-device pairs are terminal fail-closed | same as strict | no implicit device migration/repair is permitted | planned unit/property in `bd-3v0.18.5`; current anchors: `ft_device::ensure_same_device`, conformance case `device_mismatch_fail_closed` | adversarial checks stress incompatible device pairs and confirm stable mismatch diagnostics with replayable reason codes | `dispatch_key/strict:device_mismatch_fail_closed`, `dispatch_key/hardened:device_mismatch_fail_closed` | forbidden |
| `BACKEND-SCOPE-008` | `FTP2C007-B08` | dispatch key domain includes backend families outside current CPU-focused packet scope | unsupported backend families remain explicit gaps and are rejected fail-closed | `FT-I6` | unsupported backend domains must not execute | same as strict | unknown/non-scoped backend families are terminal with explicit gap markers | deferred to downstream packet closure beads (`bd-3v0.18.3`..`bd-3v0.18.9`) | differential/adversarial backend-expansion probes are mandatory before closure; current rows preserve explicit deferred posture | candidate IDs: `dispatch_key/strict:non_cpu_backend_key_candidate`, `dispatch_key/hardened:non_cpu_backend_key_candidate` | deferred_with_gap_id (`GAP-DISPATCH-007-BACKEND-DOMAIN`) |

## Contract Violation Logging Requirements

Every packet-007 contract violation event must include:
- `event_type` (contract ID + invariant class)
- `scenario_id`
- `packet_id` (`FT-P2C-007`)
- `mode`
- `seed`
- `reason_code`
- `artifact_refs`
- `replay_command`
- `env_fingerprint`
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

Anchors:
- `crates/ft-device/src/lib.rs`
- `crates/ft-dispatch/src/lib.rs`
- `crates/ft-conformance/src/lib.rs`
- `crates/ft-conformance/src/logging.rs`
- `crates/ft-conformance/fixtures/dispatch_key_cases.json`
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`

## Traceability Crosswalk (Execution Beads)

- unit/property suite realization and structured logging enforcement: `bd-3v0.18.5`
- differential/metamorphic/adversarial packet checks: `bd-3v0.18.6`
- e2e scripts + replay/forensics envelopes: `bd-3v0.18.7`
- optimization/isomorphism proof and drift guard: `bd-3v0.18.8`
- final evidence pack + RaptorQ sidecar/decode closure: `bd-3v0.18.9`

## N/A Cross-Cutting Validation Note

This artifact update is docs/planning only for packet subtask B.
Execution evidence is deferred to:
- `bd-3v0.18.5` (unit/property + structured logs)
- `bd-3v0.18.6` (differential/metamorphic/adversarial)
- `bd-3v0.18.7` (e2e/replay/forensics logging)
