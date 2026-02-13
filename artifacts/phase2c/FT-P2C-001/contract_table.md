# FT-P2C-001 — Contract Table

## Tensor Core Contract (Scoped Vertical Slice)

| Contract ID | Input | Output | Error | Invariant |
|---|---|---|---|---|
| `TENSOR-META-001` | shape + dtype + device | `TensorMeta` | `RankStrideMismatch` | `shape.len() == strides.len()` |
| `TENSOR-COMPAT-001` | lhs/rhs tensors | compatible verdict | `DTypeMismatch`, `DeviceMismatch` | binary kernels require identical dtype/device |
| `TENSOR-VERSION-001` | immutable input tensor | derived tensor with incremented version | none | out-of-place ops bump version on output |
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

