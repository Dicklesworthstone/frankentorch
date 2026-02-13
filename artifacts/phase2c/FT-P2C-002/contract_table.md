# FT-P2C-002 — Contract Table

## Dispatch Key Contract (Scoped)

| Contract ID | Input | Output | Error | Invariant |
|---|---|---|---|---|
| `DISPATCH-KEY-001` | keyset bits | validated `DispatchKeySet` | `UnknownBits` | unknown bits are fail-closed |
| `DISPATCH-KEY-002` | keyset | highest-priority type key | `EmptySet`, `NoTypeKey` | stable priority order (`AutogradCPU > Composite* > CPU > BackendSelect`) |
| `DISPATCH-KEY-003` | keyset | highest-priority backend key | `EmptySet`, `NoBackendKey` | backend key must resolve for executable path |
| `DISPATCH-KEY-004` | op + mode + tensors + keyset | `DispatchDecision` + output tensor | `IncompatibleSet`, `Kernel` | strict forbids composite fallback; hardened allows bounded fallback |
| `DISPATCH-KEY-005` | autograd-required op | `AutogradCPU` route | `IncompatibleSet` | requires-grad route emits autograd key evidence |

## Strict vs Hardened Policy

| Mode | Behavior |
|---|---|
| strict | no composite/backend-select fallback; route must be directly executable |
| hardened | allows bounded fallback to backend key when selected key is composite/backend-select |

## Behavioral Isomorphism Notes

1. Keyset algebra (`add/remove/union/intersection`) is deterministic and side-effect free.
2. Kernel math behavior remains in `ft-kernel-cpu`; this packet modifies only route selection and evidence shape.
3. Fail-closed on unknown/incompatible keysets prevents silent drift.
