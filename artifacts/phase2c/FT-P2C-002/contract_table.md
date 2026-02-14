# FT-P2C-002 — Contract Table + Strict/Hardened Invariant Spec

Packet: Dispatch key model  
Dependencies: `bd-3v0.13.1` behavior ledger + fixture manifest seed map

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

| Contract ID | Behavior ID | Preconditions | Postconditions | Invariant class | Strict semantics | Hardened semantics | Fail-closed boundary | Unit/property mapping | Differential/adversarial intent | E2E scenario IDs | Drift posture |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `DISPATCH-KEYSET-001` | `FTP2C002-B04` | raw keyset bitmask has only recognized dispatch bits | validated `DispatchKeySet` emitted | `FT-I5`, `FT-I6` | unknown bits rejected with `UnknownBits` | same fail-closed behavior | any unknown bitmask is terminal error (no repair) | `ft_dispatch::unknown_bits_fail_closed` | adversarial raw-bitmask corruption must report deterministic error taxonomy | candidate: `dispatch_key/strict:unknown_bits_mask_candidate`, `dispatch_key/hardened:unknown_bits_mask_candidate` | forbidden |
| `DISPATCH-PRIORITY-002` | `FTP2C002-B01` | keyset contains at least one type key | highest-priority type key resolved deterministically | `FT-I1`, `FT-I6` | priority order fixed (`AutogradCPU > CompositeExplicitAutograd > CompositeImplicitAutograd > CPU > BackendSelect`) | same | empty/non-type sets fail (`EmptySet`/`NoTypeKey`) | `ft_dispatch::priority_resolution_prefers_autograd_cpu` | compare local/oracle selected type key for strict + hardened fixtures | `dispatch_key/strict:strict_autograd_route`, `dispatch_key/hardened:strict_autograd_route` | forbidden |
| `DISPATCH-BACKEND-003` | `FTP2C002-B02` | keyset contains executable backend key | backend key resolved (`CPU`) | `FT-I1`, `FT-I6` | backend key required for executable route | same | missing backend fails (`EmptySet`/`NoBackendKey`) | `ft_dispatch::backend_priority_returns_cpu` | verify backend-key parity and kernel identity in differential suite | `dispatch_key/strict:strict_cpu_route`, `dispatch_key/hardened:strict_cpu_route` | forbidden |
| `DISPATCH-MODE-004` | `FTP2C002-B03` | selected type key may be composite/backend-select | dispatch decision + output tensor + fallback flag | `FT-I3`, `FT-I6` | composite/backend-select fallback rejected with explicit error | bounded fallback to backend key allowed and logged | unknown or incompatible keyset never downgraded to warning | `ft_dispatch::strict_mode_rejects_composite_fallback`, `ft_dispatch::hardened_mode_allows_composite_fallback` | strict must emit expected error; hardened must emit parity output + fallback evidence | `dispatch_key/strict:composite_route_mode_split`, `dispatch_key/hardened:composite_route_mode_split` | allowlisted_hardened_only (`dispatch.composite_backend_fallback`) |
| `DISPATCH-AUTOGRAD-005` | `FTP2C002-B01` | binary op has `requires_grad=true` | selected key includes autograd route semantics | `FT-I3`, `FT-I6` | autograd route required when grad path requested | same | incompatible autograd/backend combination fails closed | `ft_dispatch::dispatch_returns_kernel_metadata` + conformance autograd route case | adversarial incompatible keyset (`AutogradCPU` without `CPU`) tracked for differential fail-closed validation | candidate: `dispatch_key/strict:incompatible_autograd_without_cpu_candidate`, `dispatch_key/hardened:incompatible_autograd_without_cpu_candidate` | forbidden |
| `DISPATCH-KERNEL-006` | `FTP2C002-B01`, `FTP2C002-B02` | resolved route is executable | kernel output parity and decision evidence are deterministic | `FT-I1`, `FT-I3` | output + key metadata must match expected fixture contract | same output contract; fallback flag may differ only where explicitly allowlisted | kernel/route mismatch treated as parity violation | `ft_dispatch::dispatch_returns_kernel_metadata`, `ft_conformance::strict_dispatch_conformance_is_green`, `ft_conformance::hardened_dispatch_conformance_is_green` | oracle comparator verifies output, selected key, backend key, kernel, fallback flag | all listed `dispatch_key/*` scenarios in fixture manifest | forbidden except explicit hardened allowlist |
| `DISPATCH-SCOPE-007` | `FTP2C002-B06` | upstream key domain includes non-CPU/dynamic keys outside scoped slice | scoped packet remains CPU + autograd CPU only | `FT-I6` | unsupported keys must fail closed | same | out-of-scope keys are not silently coerced | deferred (no direct unit fixture yet) | deferred adversarial probes to `FT-P2C-007` backend expansion | candidate: `dispatch_key/strict:non_cpu_backend_key_candidate`, `dispatch_key/hardened:non_cpu_backend_key_candidate` | deferred_with_gap_id (`GAP-DISPATCH-001`) |

## Contract Violation Logging Requirements

Every dispatch-contract violation event must include:
- `event_type` (contract ID + invariant class)
- `scenario_id`
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

Anchors:
- `crates/ft-conformance/src/lib.rs:1635`
- `crates/ft-conformance/src/lib.rs:1732`
- `crates/ft-conformance/src/logging.rs:11`
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`

## N/A Cross-Cutting Validation Note

This artifact update is docs/planning only for packet subtask B.
Execution evidence is deferred to:
- `bd-3v0.13.5` (unit/property)
- `bd-3v0.13.6` (differential/metamorphic/adversarial)
- `bd-3v0.13.7` (e2e/logging)
