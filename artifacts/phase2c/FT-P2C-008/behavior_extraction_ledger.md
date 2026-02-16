# FT-P2C-008 — Behavior Extraction Ledger

Packet: NN module/state contract first-wave  
Legacy anchor map: `artifacts/phase2c/FT-P2C-008/legacy_anchor_map.md`

## Behavior Families (Nominal, Edge, Adversarial)

| Behavior ID | Path class | Legacy anchor family | Strict expectation | Hardened expectation | Candidate unit/property assertions | E2E scenario seed(s) |
|---|---|---|---|---|---|---|
| `FTP2C008-B01` | nominal | `register_parameter` / `register_buffer` validation and registration paths | invalid names/types are rejected; valid entries are registered in deterministic namespace | same outward validation result; no silent fallback registration | `ft_nn::register_parameter_rejects_invalid_name`, `ft_nn::register_buffer_tracks_persistence_flag` | `ft_p2c_008/nn_state/strict:register_paths`=`11942008826435908911`, `ft_p2c_008/nn_state/hardened:register_paths`=`6238485149855313380` |
| `FTP2C008-B02` | nominal | `state_dict` + `_save_to_state_dict` recursive export | exported keys/metadata are deterministic and include parameters + persistent buffers only | same outward contract; optional hardened diagnostics may annotate non-persistent buffer omission without changing keys | `ft_nn::state_dict_includes_parameters_and_persistent_buffers`, `ft_nn::state_dict_nested_prefixes_are_stable` | `ft_p2c_008/nn_state/strict:state_export`=`9105792222718758034`, `ft_p2c_008/nn_state/hardened:state_export`=`17684406980149178263` |
| `FTP2C008-B03` | nominal | `children` / `named_children` / `modules` / `named_modules` traversal | recursion order and prefix composition are deterministic and duplicate-safe | same | `ft_nn::named_modules_prefix_order_is_stable`, `ft_nn::children_iteration_matches_registration_order` | `ft_p2c_008/nn_state/strict:module_traversal`=`14389331970150071528`, `ft_p2c_008/nn_state/hardened:module_traversal`=`1029188525253192439` |
| `FTP2C008-B04` | nominal | `train(mode)` and `eval()` propagation | boolean mode validation and recursive propagation must be deterministic for all descendants | same | `ft_nn::train_propagates_recursively`, `ft_nn::eval_is_train_false_alias` | `ft_p2c_008/nn_state/strict:mode_propagation`=`7078574734342126786`, `ft_p2c_008/nn_state/hardened:mode_propagation`=`12971941327799640879` |
| `FTP2C008-B05` | edge (load strictness) | `load_state_dict` + `_load_from_state_dict` mismatch accounting | missing/unexpected/shape-incompatible entries fail closed in strict mode with deterministic error envelope | bounded non-strict load may proceed but must surface deterministic incompatibility telemetry | `ft_nn::load_state_dict_strict_rejects_unexpected_keys`, `ft_nn::load_state_dict_strict_rejects_missing_keys`, `ft_nn::load_state_dict_rejects_shape_mismatch` | `ft_p2c_008/nn_state/strict:load_mismatch`=`4706969904513634549`, `ft_p2c_008/nn_state/hardened:load_mismatch`=`17162705171385559996` |
| `FTP2C008-B06` | edge (prefix compatibility) | `consume_prefix_in_state_dict_if_present` compatibility helper | prefixed keys require explicit normalization before strict load; no implicit relaxed matching | hardened compatibility path may apply deterministic prefix normalization step prior to strictness evaluation | `ft_nn::prefix_consumption_maps_ddp_state_dict_keys`, `ft_nn::strict_load_after_prefix_consumption_is_clean` | `ft_p2c_008/nn_state/strict:prefix_normalization`=`15057396828786404366`, `ft_p2c_008/nn_state/hardened:prefix_normalization`=`4394520428593552581` |
| `FTP2C008-B07` | edge (hook boundaries) | state-dict pre/post hooks and load pre/post hooks | hooks execute in deterministic order and cannot silently violate strict compatibility outcomes | hardened path may use allowlisted hook-driven sanitization, but resulting keys/errors must remain auditable | `ft_nn::state_dict_hooks_fire_in_order`, `ft_nn::load_state_dict_pre_post_hooks_emit_trace` | `ft_p2c_008/nn_state/strict:hook_paths`=`7826767337537184328`, `ft_p2c_008/nn_state/hardened:hook_paths`=`10063960559146328173` |
| `FTP2C008-B08` | edge (device/dtype transfer) | `_apply` + `to/cpu/cuda/float/double/half/bfloat16` | transfer/cast pathways preserve module-tree and state identity invariants or fail closed on unsupported requests | hardened may permit bounded diagnostics and safe coercion gates without changing visible module-state correctness | `ft_nn::module_to_preserves_keyset_and_shapes`, `ft_nn::module_apply_recurse_toggle_is_respected` | `ft_p2c_008/nn_state/strict:transfer_paths`=`15709423042171108087`, `ft_p2c_008/nn_state/hardened:transfer_paths`=`5814065865901529136` |
| `FTP2C008-B09` | adversarial | malformed state payloads and incompatible assign pathways | incompatible state payload must fail closed with deterministic error classification | hardened non-strict/assign paths are bounded and must still reject incompatible tensor shapes/types | `ft_nn::load_state_dict_assign_rejects_incompatible_tensor_shape`, `ft_nn::load_state_dict_rejects_non_dict_payload` | `ft_p2c_008/nn_state/strict:adversarial_state_payload`=`3668047182098084497`, `ft_p2c_008/nn_state/hardened:adversarial_state_payload`=`14162254806718873162` |

## Logging Field Expectations by Behavior Family

Mandatory deterministic replay fields (all module/state families):
- `suite_id`
- `scenario_id`
- `fixture_id`
- `packet_id`
- `mode`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `outcome`
- `reason_code`

NN module/state additions:
- `module_path`
- `state_key`
- `state_key_kind` (`parameter`, `buffer`, `extra_state`)
- `persistent_flag`
- `strict_flag`
- `assign_flag`
- `missing_keys`
- `unexpected_keys`
- `incompatible_shapes`
- `hook_trace`
- `prefix_normalization_applied`
- `training_flag_transition`

Anchors:
- `legacy_pytorch_code/pytorch/torch/nn/modules/module.py`
- `legacy_pytorch_code/pytorch/torch/nn/modules/utils.py`
- `legacy_pytorch_code/pytorch/test/test_nn.py`
- `legacy_pytorch_code/pytorch/test/nn/test_load_state_dict.py`
- `artifacts/phase2c/ESSENCE_EXTRACTION_LEDGER_V1.md`

## N/A Cross-Cutting Validation Note

This ledger is docs/planning only for packet subtask A (`bd-3v0.19.1`).  
Execution-evidence ownership is carried by downstream packet beads:
- contract/invariant closure: `bd-3v0.19.2`
- security/compatibility threat model: `bd-3v0.19.3`
- implementation boundaries: `bd-3v0.19.4`
- unit/property + structured logging: `bd-3v0.19.5`
- differential/metamorphic/adversarial validation: `bd-3v0.19.6`
- e2e scripts + replay/forensics logging: `bd-3v0.19.7`
- optimization/isomorphism proof: `bd-3v0.19.8`
- final evidence pack + RaptorQ closure: `bd-3v0.19.9`
