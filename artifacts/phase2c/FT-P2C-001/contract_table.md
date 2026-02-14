# FT-P2C-001 — Contract Table

## Tensor Core Contract (Scoped Vertical Slice)

| Contract ID | Input | Output | Error | Invariant |
|---|---|---|---|---|
| `TENSOR-META-001` | shape + dtype + device | `TensorMeta` | `RankStrideMismatch` | `shape.len() == strides.len()` |
| `TENSOR-META-002` | `(shape,strides,storage_offset,index)` | linear storage index | bounds/overflow errors | indexing is overflow-safe and fail-closed |
| `TENSOR-META-003` | valid metadata case + `storage_offset + 1` transform | shifted linear index | none | metamorphic invariant: `linear_index' = linear_index + 1` while `numel`/`contiguous` stay unchanged |
| `TENSOR-META-004` | legacy oracle request for metadata case | oracle observation | guarded skip or oracle stderr | resource guard fail-closed when required backing elements exceed safe cap |
| `TENSOR-COMPAT-001` | lhs/rhs tensors | compatible verdict | `DTypeMismatch`, `DeviceMismatch` | binary kernels require identical dtype/device |
| `TENSOR-VERSION-001` | immutable input tensor | derived tensor with incremented version | none | out-of-place ops bump version on output |
| `TENSOR-ALIAS-001` | tensor + alias offset | alias tensor view | metadata validation error | alias view preserves storage identity and version |
| `AUTOGRAD-DAC-001` | DAG rooted at node `r` | deterministic `BackwardReport` | `UnknownNode` | reverse-topological replay is deterministic |
| `DISPATCH-001` | binary op + mode + tensors | `DispatchOutcome` | `Kernel` | kernel selection is explicit and logged |

## Strict vs Hardened Policy

| Mode | Behavior in this slice |
|---|---|
| strict | exact scalar op semantics with fail-fast validation |
| hardened | same math semantics; defensive checks retained; no silent repair |

## Deferred (Explicit)

- symbolic-shape parity (`sym_*`) is deferred to `FT-P2C-002/003`.
- storage alias/view graph parity is deferred to `FT-P2C-004+`.
