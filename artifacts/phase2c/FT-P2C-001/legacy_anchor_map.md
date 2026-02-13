# FT-P2C-001 — Legacy Anchor Map

Packet: Tensor metadata + storage core
Legacy root: `legacy_pytorch_code/pytorch`
Primary file: `c10/core/TensorImpl.h`

## Extracted Anchors (Exact)

| Legacy path | Line anchor | Symbol | Porting relevance |
|---|---:|---|---|
| `c10/core/TensorImpl.h` | 162 | `AutogradMetaInterface` | grad metadata and requires-grad contract surface |
| `c10/core/TensorImpl.h` | 329 | `VariableVersion` | version-counter semantics for in-place safety |
| `c10/core/TensorImpl.h` | 470-505 | dtype/storage uninitialized state notes | explicit migration warning; avoid recreating legacy ephemeral states |
| `c10/core/TensorImpl.h` | 618-809 | `sizes()/strides()/sym_*` | canonical metadata access semantics |
| `c10/core/TensorImpl.h` | 745-776 | `storage_offset()` / `sym_storage_offset()` | offset semantics and symbolic guard behavior |
| `c10/core/TensorImpl.h` | 1060-1087 | `storage()/unsafe_storage()` | guarded storage access policy |
| `c10/core/TensorImpl.h` | 1290-1301 | `device()` / `device_default()` | device invariant and undefined-tensor guard |
| `c10/core/TensorImpl.h` | 2894 | `Storage storage_` | storage ownership field anchor |
| `c10/core/TensorImpl.h` | 2920 | `autograd_meta_` | nullable autograd metadata optimization |
| `c10/core/TensorImpl.h` | 2925 | `version_counter_` | version counter always-available invariant |
| `c10/core/TensorImpl.h` | 2929-2936 | `sizes_and_strides_`, `storage_offset_`, `numel_` | metadata core field set |
| `c10/core/TensorImpl.h` | 2940 | `data_type_` | dtype invariant linked to storage |
| `c10/core/TensorImpl.h` | 2954 | `device_opt_` | optional device (undefined-tensor only) |

## Implemented Mapping in This Pass

| Rust crate | File | Mapping |
|---|---|---|
| `ft-core` | `crates/ft-core/src/lib.rs` | `TensorMeta`, `ScalarTensor`, compatibility checks, contiguous stride logic |
| `ft-autograd` | `crates/ft-autograd/src/lib.rs` | deterministic backward replay and gradient ledger steps |
| `ft-runtime` | `crates/ft-runtime/src/lib.rs` | evidence ledger + durability envelope model |

