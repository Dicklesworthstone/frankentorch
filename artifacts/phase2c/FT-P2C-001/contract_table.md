# FT-P2C-001 — Contract Table + Strict/Hardened Invariant Spec

Packet: Tensor metadata + storage core  
Dependencies: `bd-3v0.12.1` behavior ledger and fixture manifest

## Machine-Checkable Contract Row Schema

Each contract row is considered complete only if it defines:
- preconditions
- postconditions
- invariant class ID(s)
- strict-mode semantics
- hardened-mode semantics
- fail-closed boundary decision
- unit/property test mapping
- differential/metamorphic/adversarial intent
- e2e scenario ID mapping
- drift posture (`forbidden`, `allowlisted_hardened_only`, `deferred_with_gap_id`)

## Contract Rows

| Contract ID | Behavior ID | Preconditions | Postconditions | Invariant class | Strict semantics | Hardened semantics | Fail-closed boundary | Unit/property mapping | Differential/adversarial intent | E2E scenario IDs | Drift posture |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `TENSOR-META-001` | `FTP2C001-B02` | `shape.len == strides.len`; metadata validates | valid `TensorMeta` emitted | `FT-I1`, `FT-I2` | reject invalid metadata | same as strict | rank/stride mismatch and overflow are hard errors | `ft_core::shape_builds_contiguous_strides`, `ft_core::custom_strides_validate_and_index_into_storage` | compare local/oracle `numel`, `contiguous`, index behavior; include metamorphic offset-shift checks | `tensor_meta/strict:contiguous_basic_index`, `tensor_meta/hardened:contiguous_basic_index` | forbidden |
| `TENSOR-META-002` | `FTP2C001-B04` | malformed rank/stride inputs | no tensor produced; typed error | `FT-I5` | fail closed (`RankStrideMismatch`) | fail closed (`RankStrideMismatch`) | unknown/invalid metadata never repaired | `ft_core::index_rank_and_bounds_are_guarded` + invalid fixture checks | adversarial invalid fixture must produce fail-closed parity status | `tensor_meta/strict:invalid_rank_stride_mismatch`, `tensor_meta/hardened:invalid_rank_stride_mismatch` | forbidden |
| `TENSOR-META-003` | `FTP2C001-B05` | storage offset/index multiplication may overflow | no index emitted; typed overflow error | `FT-I5` | fail closed (`StrideOverflow`/`StorageOffsetOverflow`) | fail closed (`StrideOverflow`/`StorageOffsetOverflow`) | overflow branches never downgraded to warning | `ft_core::custom_strides_validate_and_index_into_storage` + invalid overflow fixture | adversarial overflow case must produce deterministic mismatch/error taxonomy | `tensor_meta/strict:invalid_storage_offset_overflow`, `tensor_meta/hardened:invalid_storage_offset_overflow` | forbidden |
| `TENSOR-ALIAS-001` | `FTP2C001-B03` | valid base tensor + alias offset | alias view shares storage identity | `FT-I2`, `FT-I4` | alias semantics deterministic | same as strict | invalid alias metadata rejected | `ft_core::alias_view_shares_storage_identity` | verify alias storage identity in differential tensor-meta checks | `tensor_meta/strict:scalar_offset_index`, `tensor_meta/hardened:scalar_offset_index` | forbidden |
| `TENSOR-VERSION-001` | `FTP2C001-B03` | valid source tensor; out-of-place op | output version increments; storage identity differs | `FT-I4` | deterministic version bump required | same as strict | missing version bump is contract violation | `ft_core::out_of_place_result_gets_new_storage_and_version_bump` | detect version/identity drift as parity mismatch | `scalar_dac/strict:add_basic`, `scalar_dac/hardened:add_basic` | forbidden |
| `TENSOR-COMPAT-001` | `FTP2C001-B06` | lhs/rhs dtype/device comparable | compatible pair passes; mismatch errors | `FT-I6` | fail closed on mismatch | fail closed on mismatch | no silent coercion in this packet | candidate tests for explicit mismatch branches (pending) | add adversarial mismatch differential checks once fixture exists | candidate IDs: `tensor_meta/strict:compat_device_mismatch_candidate`, `tensor_meta/hardened:compat_device_mismatch_candidate` | deferred_with_gap_id (`GAP-UX-001`) |
| `AUTOGRAD-DAC-001` | `FTP2C001-B01` | DAG built from scalar add/mul nodes | deterministic backward report and gradients | `FT-I1`, `FT-I3` | deterministic replay contract enforced | same math contract; bounded policy instrumentation only | unknown node/dependency mismatch fails closed | `ft_autograd::add_backward_matches_expected_gradient`, `ft_autograd::mul_backward_matches_expected_gradient` | oracle output/gradient comparators must match per case | `scalar_dac/strict:add_basic`, `scalar_dac/hardened:mul_basic` | forbidden |
| `DISPATCH-001` | `FTP2C001-B01` | valid binary op + tensors | explicit dispatch decision + outcome | `FT-I6` | strict route must not use composite fallback | hardened may use bounded fallback when configured | unknown keysets/incompatible sets fail closed | `ft_dispatch::dispatch_returns_kernel_metadata` | differential checks compare selected key and fallback behavior | `dispatch_key/strict:strict_autograd_route`, `dispatch_key/hardened:strict_autograd_route` | allowlisted_hardened_only for specific drift IDs |

## Contract Violation Logging Requirements

Every violation event must include:
- `event_type` (contract ID + invariant class)
- `scenario_id`
- `mode`
- `seed`
- `reason_code`
- `artifact_refs`
- `replay_command`
- `env_fingerprint`

Anchor references:
- `crates/ft-conformance/src/logging.rs:11`
- `crates/ft-conformance/src/lib.rs:1943`
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`

## N/A Cross-Cutting Validation Note

This artifact update is docs/planning only for packet subtask B.
Execution evidence is deferred to:
- `bd-3v0.12.5` (unit/property)
- `bd-3v0.12.6` (differential/metamorphic/adversarial)
- `bd-3v0.12.7` (e2e/logging)
