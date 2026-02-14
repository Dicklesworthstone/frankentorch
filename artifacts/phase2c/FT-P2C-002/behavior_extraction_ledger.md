# FT-P2C-002 — Behavior Extraction Ledger

Packet: Dispatch key model  
Legacy anchor map: `artifacts/phase2c/FT-P2C-002/legacy_anchor_map.md`

## Behavior Families (Nominal, Edge, Adversarial)

| Behavior ID | Path class | Legacy anchor family | Strict expectation | Hardened expectation | Candidate unit/property assertions | E2E scenario seed(s) |
|---|---|---|---|---|---|---|
| `FTP2C002-B01` | nominal | `DispatchKeySet::highestPriorityTypeId` (`c10/core/DispatchKeySet.h`) | autograd-required route resolves `AutogradCPU`, no fallback | same route, no fallback | `ft_dispatch::priority_resolution_prefers_autograd_cpu`, `ft_dispatch::dispatch_returns_kernel_metadata`, `ft_conformance::strict_dispatch_conformance_is_green` | `dispatch_key/strict:strict_autograd_route`=`12780237016247668875`, `dispatch_key/hardened:strict_autograd_route`=`456065680046437289` |
| `FTP2C002-B02` | nominal | backend projection helper (`highestPriorityBackendTypeId`) | CPU-only route resolves `CPU` backend key | same | `ft_dispatch::backend_priority_returns_cpu`, `ft_dispatch::dispatch_returns_kernel_metadata`, `ft_conformance::hardened_dispatch_conformance_is_green` | `dispatch_key/strict:strict_cpu_route`=`6654506012862553729`, `dispatch_key/hardened:strict_cpu_route`=`7720564810062027209` |
| `FTP2C002-B03` | edge (mode split) | composite dispatch policy in key priority + routing | composite/backend-select route is fail-closed error | bounded fallback to executable backend key with explicit evidence flag | `ft_dispatch::strict_mode_rejects_composite_fallback`, `ft_dispatch::hardened_mode_allows_composite_fallback` | `dispatch_key/strict:composite_route_mode_split`=`14228129716249401336`, `dispatch_key/hardened:composite_route_mode_split`=`2146157517907283417` |
| `FTP2C002-B04` | adversarial | raw keyset bitmask parser (`from_bits_checked`) | unknown bits fail closed (`UnknownBits`) | same fail-closed behavior | `ft_dispatch::unknown_bits_fail_closed` | candidate seeds: `dispatch_key/strict:unknown_bits_mask_candidate`=`201313001`, `dispatch_key/hardened:unknown_bits_mask_candidate`=`201313002` |
| `FTP2C002-B05` | adversarial | keyset compatibility validator (`validate_for_scalar_binary`) | incompatible keyset (`AutogradCPU` without `CPU`) is rejected | same fail-closed behavior | candidate unit assertion for `DispatchKeyError::IncompatibleSet`; differential adversarial check to be attached in `bd-3v0.13.6` | candidate seeds: `dispatch_key/strict:incompatible_autograd_without_cpu_candidate`=`201313003`, `dispatch_key/hardened:incompatible_autograd_without_cpu_candidate`=`201313004` |
| `FTP2C002-B06` | deferred parity edge | expanded key domains in upstream `DispatchKey` enum | non-CPU/dynamic key families are out-of-scope for this packet and must fail closed if encountered | same | deferred to packet chain (`FT-P2C-007` backend expansion, `FT-P2C-003` schema-ingested op routing) | candidate seeds: `dispatch_key/strict:non_cpu_backend_key_candidate`=`201313005`, `dispatch_key/hardened:non_cpu_backend_key_candidate`=`201313006` |

## Logging Field Expectations by Behavior Family

Mandatory deterministic replay fields (all behavior families):
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

Dispatch-family additions:
- `selected_key`
- `backend_key`
- `keyset_bits`
- `fallback_used`

Anchors:
- `crates/ft-conformance/src/lib.rs:1635`
- `crates/ft-conformance/src/lib.rs:1732`
- `crates/ft-conformance/src/logging.rs:11`
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`

## N/A Cross-Cutting Validation Note

This ledger is docs/planning only for packet subtask A.
Execution evidence is carried by downstream packet beads:
- unit/property: `bd-3v0.13.5`
- differential/adversarial: `bd-3v0.13.6`
- e2e/logging: `bd-3v0.13.7`
