# FT-P2C-001 — Legacy Anchor Map

Packet: Tensor metadata + storage core
Legacy root: `legacy_pytorch_code/pytorch`
Primary file: `c10/core/TensorImpl.h`

## Extracted Anchors (Exact)

| Legacy path | Line anchor | Symbol | Porting relevance | Confidence |
|---|---:|---|---|
| `c10/core/TensorImpl.h` | 162 | `AutogradMetaInterface` | grad metadata and requires-grad contract surface | high |
| `c10/core/TensorImpl.h` | 329 | `VariableVersion` | version-counter semantics for in-place safety | high |
| `c10/core/TensorImpl.h` | 470-505 | dtype/storage uninitialized state notes | explicit migration warning; avoid recreating legacy ephemeral states | medium |
| `c10/core/TensorImpl.h` | 618-809 | `sizes()/strides()/sym_*` | canonical metadata access semantics | high |
| `c10/core/TensorImpl.h` | 745-776 | `storage_offset()` / `sym_storage_offset()` | offset semantics and symbolic guard behavior | high |
| `c10/core/TensorImpl.h` | 1060-1087 | `storage()/unsafe_storage()` | guarded storage access policy | medium |
| `c10/core/TensorImpl.h` | 1290-1301 | `device()` / `device_default()` | device invariant and undefined-tensor guard | high |
| `c10/core/TensorImpl.h` | 2894 | `Storage storage_` | storage ownership field anchor | high |
| `c10/core/TensorImpl.h` | 2920 | `autograd_meta_` | nullable autograd metadata optimization | medium |
| `c10/core/TensorImpl.h` | 2925 | `version_counter_` | version counter always-available invariant | high |
| `c10/core/TensorImpl.h` | 2929-2936 | `sizes_and_strides_`, `storage_offset_`, `numel_` | metadata core field set | high |
| `c10/core/TensorImpl.h` | 2940 | `data_type_` | dtype invariant linked to storage | high |
| `c10/core/TensorImpl.h` | 2954 | `device_opt_` | optional device (undefined-tensor only) | medium |

## Implemented Mapping in This Pass

| Rust crate | File | Mapping |
|---|---|---|
| `ft-core` | `crates/ft-core/src/lib.rs` | `TensorMeta`, `ScalarTensor`, compatibility checks, contiguous stride logic |
| `ft-autograd` | `crates/ft-autograd/src/lib.rs` | deterministic backward replay and gradient ledger steps |
| `ft-runtime` | `crates/ft-runtime/src/lib.rs` | evidence ledger + durability envelope model |

## Confidence Notes and Undefined Regions

- `high` confidence anchors map directly to immutable structural fields or strongly typed APIs in legacy source.
- `medium` confidence anchors require policy interpretation during porting (notably optional/unsafe storage access, nullable autograd metadata, and legacy uninitialized notes).
- Undefined/deferred zones for this packet:
  - symbolic shape full parity (`sym_*` behavior expansion) is deferred to later packets.
  - alias graph semantics beyond scalar/tensor-meta slice are deferred.
