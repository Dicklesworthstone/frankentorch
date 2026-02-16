# FT-P2C-008 — Security + Compatibility Threat Model

Packet: NN module/state contract first-wave  
Scope owner bead: `bd-3v0.19.3`

## Boundary and Assets

In-scope asset surfaces:
- module registration semantics (`register_parameter`, `register_buffer`) and namespace integrity
- module state export/load contract (`state_dict`, `_save_to_state_dict`, `load_state_dict`, `_load_from_state_dict`)
- module traversal and mode propagation (`children`, `named_modules`, `train`, `eval`)
- compatibility helpers and hook boundaries (`consume_prefix_in_state_dict_if_present`, pre/post state hooks)
- device/dtype transfer pathways for module state (`_apply`, `to`, `cpu`, `cuda`, dtype casts)
- packet artifacts under `artifacts/phase2c/FT-P2C-008/`

Out-of-scope (explicit, non-silent):
- full optimizer-state and scheduler-state surface beyond module first-wave packet scope
- full PyTorch module ecosystem closure (distributed wrappers, quantization stacks, export-specific module wrappers)
- backend-domain expansion beyond currently scoped compatibility envelope

## Compatibility Envelope and Mode-Split Fail-Closed Rules

| Boundary | Strict mode | Hardened mode | Fail-closed rule |
|---|---|---|---|
| registration validation | exact name/type/ownership checks, deterministic failures | same outward registration contract | invalid names/types/ownership are terminal |
| state export keyset | deterministic key/metadata contract; persistent buffers only | same exported keyset | unexpected key mutation/suppression is terminal |
| load strictness | `strict=true` requires exact compatibility | bounded `strict=false` path allowed with deterministic incompatibility reporting | incompatible shapes/types are terminal in both modes |
| prefix normalization | explicit compatible prefix normalization only | bounded auto-normalization allowed only via allowlisted policy | unknown prefix transforms are terminal |
| hook-driven mutation | hooks cannot hide strict incompatibilities | bounded hook sanitization only with deterministic trace + allowlist policy | hidden incompatibility is terminal |
| module transfer/cast | unsupported transitions fail closed | bounded diagnostics/guarded casts only under allowlist policy | unsupported transitions are terminal |
| malformed payload | non-dict/incompatible payload is terminal | same terminal outcome; bounded diagnostics allowed | malformed payload acceptance is forbidden |

## Threat Classes and Policy Response

| Threat ID | Abuse Class | Entry Vector | Strict Response | Hardened Response | Evidence/Test Hooks | Deterministic Scenario Seed(s) |
|---|---|---|---|---|---|---|
| `T008-01` | registration namespace poisoning | invalid/dotted/duplicate names or non-parameter/non-tensor payloads | fail closed with deterministic registration error taxonomy | same fail-closed behavior | `ft_nn::register_parameter_rejects_invalid_name`, `ft_nn::register_buffer_tracks_persistence_flag` | `ft_p2c_008/nn_state/strict:register_paths`=`11942008826435908911`, `ft_p2c_008/nn_state/hardened:register_paths`=`6238485149855313380` |
| `T008-02` | state_dict keyset drift | attempt to silently include/exclude keys or mutate metadata layout | fail closed on contract mismatch | same outward key contract; bounded diagnostics only | `ft_nn::state_dict_includes_parameters_and_persistent_buffers`, `ft_nn::state_dict_nested_prefixes_are_stable` | `ft_p2c_008/nn_state/strict:state_export`=`9105792222718758034`, `ft_p2c_008/nn_state/hardened:state_export`=`17684406980149178263` |
| `T008-03` | strictness bypass via missing/unexpected keys | `load_state_dict` payload with key drift and shape mismatch | strict load rejects deterministically | bounded non-strict path only with deterministic incompatibility trace | `ft_nn::load_state_dict_strict_rejects_unexpected_keys`, `ft_nn::load_state_dict_strict_rejects_missing_keys`, `ft_nn::load_state_dict_rejects_shape_mismatch` | `ft_p2c_008/nn_state/strict:load_mismatch`=`4706969904513634549`, `ft_p2c_008/nn_state/hardened:load_mismatch`=`17162705171385559996` |
| `T008-04` | prefix confusion / wrapper spoofing | prefixed checkpoint keys (`module.` variants) and malformed prefix patterns | explicit normalization required before strict evaluation | bounded auto-normalization under allowlist with telemetry | `ft_nn::prefix_consumption_maps_ddp_state_dict_keys`, `ft_nn::strict_load_after_prefix_consumption_is_clean` | `ft_p2c_008/nn_state/strict:prefix_normalization`=`15057396828786404366`, `ft_p2c_008/nn_state/hardened:prefix_normalization`=`4394520428593552581` |
| `T008-05` | hook-based compatibility tampering | pre/post state hooks mutate keys, incompatibility sets, or payloads | hook path cannot suppress strict compatibility failure | bounded hook sanitization only with deterministic hook-trace evidence | `ft_nn::state_dict_hooks_fire_in_order`, `ft_nn::load_state_dict_pre_post_hooks_emit_trace` | `ft_p2c_008/nn_state/strict:hook_paths`=`7826767337537184328`, `ft_p2c_008/nn_state/hardened:hook_paths`=`10063960559146328173` |
| `T008-06` | transfer/cast state corruption | `_apply`/`to`/dtype cast requests attempt unsupported module-state transitions | unsupported transition fails closed | bounded diagnostics/guarded cast path under allowlist policy | `ft_nn::module_to_preserves_keyset_and_shapes`, `ft_nn::module_apply_recurse_toggle_is_respected` | `ft_p2c_008/nn_state/strict:transfer_paths`=`15709423042171108087`, `ft_p2c_008/nn_state/hardened:transfer_paths`=`5814065865901529136` |
| `T008-07` | malformed payload / assign abuse | non-dict payload or incompatible `assign` path | terminal rejection with deterministic error class | same terminal rejection with bounded diagnostics | `ft_nn::load_state_dict_assign_rejects_incompatible_tensor_shape`, `ft_nn::load_state_dict_rejects_non_dict_payload` | `ft_p2c_008/nn_state/strict:adversarial_state_payload`=`3668047182098084497`, `ft_p2c_008/nn_state/hardened:adversarial_state_payload`=`14162254806718873162` |

## Adversarial Fixture and Failure-Injection E2E Plan

Current packet-adversarial priorities:
- strict vs hardened load mismatch policy split (`missing`, `unexpected`, `shape mismatch`)
- prefix normalization allowlist correctness for wrapped checkpoints
- hook-path auditability and incompatibility preservation
- transfer/cast guardrails for module-state transitions
- malformed payload and assign-path rejection taxonomy

Execution-bead ownership for closure evidence:
- unit/property + deterministic structured logs: `bd-3v0.19.5`
- differential/metamorphic/adversarial packet checks: `bd-3v0.19.6`
- e2e failure-injection and replay/forensics artifacts: `bd-3v0.19.7`

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

Packet-008-specific forensic additions:
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

Artifact chain requirements:
1. packet threat model (`artifacts/phase2c/FT-P2C-008/threat_model.md`)
2. packet contract ledger (`artifacts/phase2c/FT-P2C-008/contract_table.md`)
3. packet e2e forensics JSONL slice (owned by `bd-3v0.19.7`)
4. packet differential report/reconciliation artifacts (owned by `bd-3v0.19.6`)
5. packet final evidence and RaptorQ closure artifacts (owned by `bd-3v0.19.9`)

## Release-Gate Implications

1. Any non-allowlisted hardened recovery behavior is release-blocking.
2. Unknown/incompatible module-state paths must fail closed in both modes.
3. Hook-driven or prefix-driven compatibility adjustments must remain deterministic and fully auditable.
4. No packet-008 closure claim is valid without deterministic adversarial and e2e forensics evidence.

## N/A Cross-Cutting Validation Note

This bead output is docs/planning for packet subtask C (`bd-3v0.19.3`).
Execution evidence ownership is explicitly delegated to:
- unit/property: `bd-3v0.19.5`
- differential/metamorphic/adversarial: `bd-3v0.19.6`
- e2e/replay forensics: `bd-3v0.19.7`
- optimization/isomorphism and final packet closure: `bd-3v0.19.8`, `bd-3v0.19.9`
