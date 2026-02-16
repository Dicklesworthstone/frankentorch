# FT-P2C-008 — Contract Table + Strict/Hardened Invariant Spec

Packet: NN module/state contract first-wave  
Dependencies: `bd-3v0.19.1` behavior extraction ledger + legacy anchor map

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
| `MODULE-REGISTER-001` | `FTP2C008-B01` | module registration API receives candidate parameter/buffer name and payload | valid entries are registered in deterministic namespace; invalid entries are rejected with deterministic reason taxonomy | `FT-I1`, `FT-I2`, `FT-I6` | invalid name/type/ownership paths are terminal failures | same outward contract; no implicit name/type coercion | malformed registration input is always terminal | `ft_nn::register_parameter_rejects_invalid_name`, `ft_nn::register_buffer_tracks_persistence_flag` | differential checks compare registration acceptance/rejection sets against fixture oracle; adversarial probes mutate names (`""`, dotted names, duplicate attrs) and require stable failures | `ft_p2c_008/nn_state/strict:register_paths`, `ft_p2c_008/nn_state/hardened:register_paths` | forbidden |
| `MODULE-STATE-002` | `FTP2C008-B02` | registered module tree contains parameters, persistent buffers, and optional extra state | `state_dict` key set and metadata layout are deterministic; non-persistent buffers excluded | `FT-I1`, `FT-I2`, `FT-I3` | exported key/metadata set must exactly match scoped contract | same external state set; hardened diagnostics may annotate omission decisions without mutating key membership | incompatible serialization envelope for module state is terminal | `ft_nn::state_dict_includes_parameters_and_persistent_buffers`, `ft_nn::state_dict_nested_prefixes_are_stable` | metamorphic checks replay equivalent module trees with reordered construction and require stable key naming/output parity | `ft_p2c_008/nn_state/strict:state_export`, `ft_p2c_008/nn_state/hardened:state_export` | forbidden |
| `MODULE-TRAVERSAL-003` | `FTP2C008-B03` | module tree has nested children and duplicate references may exist | traversal APIs return deterministic iteration with prefix composition and duplicate handling semantics | `FT-I1`, `FT-I3` | traversal output/order must match scoped PyTorch-observable contract | same | traversal anomalies (missing subtree, unstable prefix map) are terminal compatibility failures | `ft_nn::named_modules_prefix_order_is_stable`, `ft_nn::children_iteration_matches_registration_order` | differential checks compare module-path enumeration against fixture map; adversarial probe uses duplicate module references to verify dedupe policy | `ft_p2c_008/nn_state/strict:module_traversal`, `ft_p2c_008/nn_state/hardened:module_traversal` | forbidden |
| `MODULE-MODE-004` | `FTP2C008-B04` | root module receives `train(mode)` or `eval()` transition request | training flag propagates recursively and deterministically across descendants | `FT-I1`, `FT-I6` | non-boolean mode inputs rejected; `eval()` exact alias for `train(false)` | same | invalid mode input is terminal; no implicit conversion | `ft_nn::train_propagates_recursively`, `ft_nn::eval_is_train_false_alias` | metamorphic checks apply repeated train/eval toggles and require idempotent terminal state and trace stability | `ft_p2c_008/nn_state/strict:mode_propagation`, `ft_p2c_008/nn_state/hardened:mode_propagation` | forbidden |
| `MODULE-LOAD-STRICT-005` | `FTP2C008-B05` | checkpoint state payload presented to `load_state_dict` with potential missing/unexpected/shape-mismatch keys | strict path fails with deterministic incompatibility envelope; non-strict path returns deterministic incompatibility set | `FT-I5`, `FT-I6` | strict load rejects any key/shape incompatibility | hardened path may permit bounded `strict=false` continuation while preserving explicit incompatibility telemetry and no silent key mutation | incompatible shape/type payload is always terminal in both modes | `ft_nn::load_state_dict_strict_rejects_unexpected_keys`, `ft_nn::load_state_dict_strict_rejects_missing_keys`, `ft_nn::load_state_dict_rejects_shape_mismatch` | differential checks verify strict-vs-hardened policy split and ensure hardened success cases still expose deterministic missing/unexpected key sets | `ft_p2c_008/nn_state/strict:load_mismatch`, `ft_p2c_008/nn_state/hardened:load_mismatch` | allowlisted_hardened_only (`nn_state.non_strict_missing_unexpected`) |
| `MODULE-PREFIX-006` | `FTP2C008-B06` | incoming state payload contains known wrapper prefixes (e.g., `module.`) | deterministic prefix normalization yields canonical key namespace before load evaluation | `FT-I2`, `FT-I6` | strict path requires explicit prefix-normalization step and then normal strict checks | hardened may apply bounded auto-normalization step with emitted telemetry before strictness evaluation | unknown/incompatible prefix transforms are terminal failures | `ft_nn::prefix_consumption_maps_ddp_state_dict_keys`, `ft_nn::strict_load_after_prefix_consumption_is_clean` | adversarial checks mutate prefix patterns and ensure only allowlisted normalization is accepted | `ft_p2c_008/nn_state/strict:prefix_normalization`, `ft_p2c_008/nn_state/hardened:prefix_normalization` | allowlisted_hardened_only (`nn_state.prefix_normalization_pre_load`) |
| `MODULE-HOOKS-007` | `FTP2C008-B07` | state-dict pre/post hooks and load pre/post hooks are registered | hook execution order and mutation envelope are deterministic and auditable | `FT-I1`, `FT-I6` | hooks cannot suppress strict incompatibility outcomes silently | hardened may allow bounded hook-driven sanitization only with deterministic trace evidence and allowlist contract | hook path that hides incompatibilities or mutates protected invariants is terminal | `ft_nn::state_dict_hooks_fire_in_order`, `ft_nn::load_state_dict_pre_post_hooks_emit_trace` | differential checks compare hook trace and resulting incompatibility sets across strict/hardened mode split | `ft_p2c_008/nn_state/strict:hook_paths`, `ft_p2c_008/nn_state/hardened:hook_paths` | allowlisted_hardened_only (`nn_state.hook_sanitization`) |
| `MODULE-TRANSFER-008` | `FTP2C008-B08` | module state receives `_apply`/`to`/cast transition request | state identity and key-shape invariants preserved under supported transitions | `FT-I1`, `FT-I3`, `FT-I6` | unsupported transfer/cast requests fail closed with deterministic diagnostics | hardened may permit bounded diagnostics/guarded casts while preserving observable state contracts | unsupported backend/device/dtype transitions are terminal | `ft_nn::module_to_preserves_keyset_and_shapes`, `ft_nn::module_apply_recurse_toggle_is_respected` | metamorphic checks compare equivalent transfer pipelines (`to(cpu)`, `cpu()`) for stable state output and compatibility traces | `ft_p2c_008/nn_state/strict:transfer_paths`, `ft_p2c_008/nn_state/hardened:transfer_paths` | allowlisted_hardened_only (`nn_state.transfer_guarded_cast`) |
| `MODULE-ADVERSARIAL-009` | `FTP2C008-B09` | payload is malformed/non-dict or uses incompatible assign pathway | load path rejects invalid payloads and incompatible assignments deterministically | `FT-I5`, `FT-I6` | malformed payload/assign mismatch is terminal | same terminal outcome for incompatible payload; hardened path may retain bounded diagnostics only | malformed payload acceptance is forbidden | `ft_nn::load_state_dict_assign_rejects_incompatible_tensor_shape`, `ft_nn::load_state_dict_rejects_non_dict_payload` | adversarial differential suite mutates payload kinds, shape vectors, and assign flags; requires stable fail taxonomy | `ft_p2c_008/nn_state/strict:adversarial_state_payload`, `ft_p2c_008/nn_state/hardened:adversarial_state_payload` | forbidden |

## Contract Violation Logging Requirements

Every packet-008 contract violation event must include:
- `event_type` (contract ID + invariant class)
- `scenario_id`
- `packet_id` (`FT-P2C-008`)
- `mode`
- `seed`
- `reason_code`
- `artifact_refs`
- `replay_command`
- `env_fingerprint`
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

Anchors:
- `artifacts/phase2c/FT-P2C-008/legacy_anchor_map.md`
- `artifacts/phase2c/FT-P2C-008/behavior_extraction_ledger.md`
- `legacy_pytorch_code/pytorch/torch/nn/modules/module.py`
- `legacy_pytorch_code/pytorch/torch/nn/modules/utils.py`
- `legacy_pytorch_code/pytorch/test/test_nn.py`
- `legacy_pytorch_code/pytorch/test/nn/test_load_state_dict.py`

## Traceability Crosswalk (Execution Beads)

- unit/property suite realization and structured logging enforcement: `bd-3v0.19.5`
- differential/metamorphic/adversarial packet checks: `bd-3v0.19.6`
- e2e scripts + replay/forensics envelopes: `bd-3v0.19.7`
- optimization/isomorphism proof and drift guard: `bd-3v0.19.8`
- final evidence pack + RaptorQ sidecar/decode closure: `bd-3v0.19.9`

## N/A Cross-Cutting Validation Note

This artifact update is docs/planning only for packet subtask B.  
Execution evidence is deferred to:
- `bd-3v0.19.5` (unit/property + structured logs)
- `bd-3v0.19.6` (differential/metamorphic/adversarial)
- `bd-3v0.19.7` (e2e/replay/forensics logging)
