# FT-P2C-008 — Legacy Anchor Map

Packet: NN module/state contract first-wave  
Legacy root: `legacy_pytorch_code/pytorch`

## Extracted Anchors (Exact)

| Legacy path | Line anchor | Symbol | Porting relevance |
|---|---:|---|---|
| `torch/nn/modules/module.py` | 529 | `Module.register_buffer(...)` | canonical buffer registration contract: name validation, tensor-or-None typing, persistence flag handling, and state participation |
| `torch/nn/modules/module.py` | 593 | `Module.register_parameter(...)` | canonical parameter registration contract: name validation, parameter-or-None typing, leaf/grad semantics |
| `torch/nn/modules/module.py` | 893 | `Module.get_extra_state(...)` | extension point for custom module state serialization |
| `torch/nn/modules/module.py` | 914 | `Module.set_extra_state(...)` | companion extension point for state load path |
| `torch/nn/modules/module.py` | 931 | `Module._apply(...)` | low-level recursive tensor transform primitive used by `to/cpu/cuda/dtype` transitions |
| `torch/nn/modules/module.py` | 1225 | `Module.to_empty(...)` | explicit empty-allocation device transition path with recursion control |
| `torch/nn/modules/module.py` | 1242 | `Module.to(...)` overload family | canonical dtype/device/non_blocking/memory_format transfer contract |
| `torch/nn/modules/module.py` | 1080 | `Module.cuda(...)` | backend/device transition wrapper over `_apply` |
| `torch/nn/modules/module.py` | 1156 | `Module.cpu(...)` | CPU transition wrapper over `_apply` |
| `torch/nn/modules/module.py` | 1181 | `Module.float(...)` | dtype cast contract for floating modules |
| `torch/nn/modules/module.py` | 2107 | `Module.register_state_dict_post_hook(...)` | post-serialization mutation/inspection hook boundary |
| `torch/nn/modules/module.py` | 2131 | `Module.register_state_dict_pre_hook(...)` | pre-serialization hook boundary |
| `torch/nn/modules/module.py` | 2144 | `Module._save_to_state_dict(...)` | parameter/buffer extraction and destination population primitive |
| `torch/nn/modules/module.py` | 2195 | `Module.state_dict(...)` | top-level deterministic state export traversal and metadata wiring |
| `torch/nn/modules/module.py` | 2285 | `Module._register_load_state_dict_pre_hook(...)` | pre-load hook extension used for compatibility rewrites/sanitization |
| `torch/nn/modules/module.py` | 2317 | `Module.register_load_state_dict_post_hook(...)` | post-load compatibility hook boundary |
| `torch/nn/modules/module.py` | 2346 | `Module._load_from_state_dict(...)` | recursive load primitive with missing/unexpected/error accumulation |
| `torch/nn/modules/module.py` | 2531 | `Module.load_state_dict(...)` | strict-vs-non-strict compatibility and incompatibility reporting contract |
| `torch/nn/modules/module.py` | 2777 | `Module.children(...)` | module-tree traversal primitive used by mode/state propagation |
| `torch/nn/modules/module.py` | 2837 | `Module.named_modules(...)` | recursive traversal with prefix composition and de-dup semantics |
| `torch/nn/modules/module.py` | 2886 | `Module.train(mode=True)` | deterministic training-flag propagation across module tree |
| `torch/nn/modules/module.py` | 2908 | `Module.eval()` | canonical `train(False)` delegation contract |
| `torch/nn/modules/utils.py` | 48 | `consume_prefix_in_state_dict_if_present(...)` | DDP/module-prefix compatibility helper required for checkpoint interoperability |
| `torch/nn/parameter.py` | 30 | `class Parameter` | parameter identity and autograd-visible state carrier semantics |
| `torch/nn/parameter.py` | 265 | `class Buffer` | buffer persistence contract and state-dict inclusion semantics |

## Behavioral Oracle Tests (Scoped)

| Legacy path | Intent |
|---|---|
| `test/test_nn.py` (`544`, `552`, `571`, `593`, `646`, `661`) | registration and state visibility contracts for buffers/parameters |
| `test/test_nn.py` (`2358`) | nested module state_dict key/metadata shape expectations |
| `test/nn/test_load_state_dict.py` (`88`, `118`, `132`, `149`, `162`) | strict mismatch behavior, prefix consumption, missing/unexpected key contract |
| `test/nn/test_load_state_dict.py` (`228`, `247`, `288`) | pre/post hooks and custom load behavior including assign path |
| `test/test_nn.py` (`1892`, `2066`) | recursive `train()/eval()` mode propagation behavior |

## Implemented/Planned Rust Mapping

| Rust crate | File | Mapping |
|---|---|---|
| `ft-api` | `crates/ft-api/src/lib.rs` | planned module/session-facing entry points that expose first-wave NN module state semantics |
| `ft-serialize` | `crates/ft-serialize/src/lib.rs` | checkpoint/state serialization durability and compatibility boundaries |
| `ft-conformance` | `crates/ft-conformance/src/lib.rs` | planned packet fixture harness for strict/hardened module-state scenarios |
| `ft-conformance` | `crates/ft-conformance/fixtures` | planned fixture family for state_dict/load_state_dict/module-traversal compatibility |

## Extraction Schema (Mandatory)

1. `packet_id`: `FT-P2C-008`
2. `legacy_paths`: `torch/nn/modules/module.py`, `torch/nn/modules/utils.py`, `torch/nn/parameter.py`
3. `legacy_symbols`: `register_buffer`, `register_parameter`, `state_dict`, `_save_to_state_dict`, `load_state_dict`, `_load_from_state_dict`, `train`, `eval`, `_apply`, `to`, `consume_prefix_in_state_dict_if_present`, `Parameter`, `Buffer`
4. `tensor_storage_contract`: parameters and persistent buffers are the canonical state carriers; non-persistent buffers are excluded from exported state
5. `dispatch_contract`: module-state operations must not alter dispatch-key ordering or route selection invariants from packets `FT-P2C-002` and `FT-P2C-007`
6. `error_contract`: strict load rejects unknown/incompatible state entries; mismatch diagnostics must remain deterministic and replayable
7. `grad_graph_contract`: parameter registration preserves grad participation boundaries (`requires_grad` and parameter identity)
8. `serialization_contract`: state export/load remains deterministic and checkpoint-compatible with packet `FT-P2C-006` durability constraints
9. `strict_mode_policy`: exact key-set and shape compatibility required for `load_state_dict(..., strict=True)`; no silent drops/mutations
10. `hardened_mode_policy`: bounded compatibility handling through explicit hook/prefix/assign pathways without altering observable state semantics
11. `sequencing_boundary`: packet-008 first wave covers registration/state traversal/load contracts only; full optimizer/module subclass breadth lands in downstream packet beads
12. `oracle_tests`: `test/test_nn.py`, `test/nn/test_load_state_dict.py`
13. `performance_sentinels`: state export/load latency tails, recursive traversal overhead, and hook-path overhead under representative module trees
14. `compatibility_risks`: incomplete module-family coverage during first wave can hide key-shape drift and hook-order divergence if not explicitly gated
15. `raptorq_artifacts`: packet parity-report sidecar + decode-proof closure remain mandatory in final evidence bead (`bd-3v0.19.9`)
