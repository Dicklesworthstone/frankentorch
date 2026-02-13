# FT-P2C-002 — Legacy Anchor Map

Packet: Dispatch key model  
Legacy root: `legacy_pytorch_code/pytorch`

## Extracted Anchors (Exact)

| Legacy path | Line anchor | Symbol | Porting relevance |
|---|---:|---|---|
| `c10/core/DispatchKey.h` | 136 | `enum class DispatchKey : uint16_t` | canonical key domain and ordering source |
| `c10/core/DispatchKeySet.h` | 167 | `class DispatchKeySet` | keyset algebra and bitset representation |
| `c10/core/DispatchKeySet.h` | 434 | `DispatchKey highestPriorityTypeId() const` | priority resolution contract |
| `c10/core/DispatchKeySet.h` | 913 | `inline DispatchKey highestPriorityBackendTypeId(DispatchKeySet ks)` | backend projection helper |
| `aten/src/ATen/Dispatch.h` | 183 | `AT_DISPATCH_SWITCH(TYPE, NAME, ...)` | dtype-driven dispatch macro architecture |
| `aten/src/ATen/Dispatch.h` | 191 | `AT_DISPATCH_CASE_FLOATING_TYPES(...)` | case macro style and key-to-kernel expansion model |

## Implemented Mapping

| Rust crate | File | Mapping |
|---|---|---|
| `ft-dispatch` | `crates/ft-dispatch/src/lib.rs` | `DispatchKey`, `DispatchKeySet`, keyset validation, priority resolvers, strict/hardened fallback routing |
| `ft-autograd` | `crates/ft-autograd/src/lib.rs` | `dispatch_scalar_binary(... requires_grad)` integration for autograd-aware key selection |
| `ft-conformance` | `crates/ft-conformance/src/lib.rs` | dispatch-route conformance family (`dispatch_key_cases.json`) |

## Extraction Schema (Mandatory)

1. `packet_id`: `FT-P2C-002`
2. `legacy_paths`: `c10/core/DispatchKey.h`, `c10/core/DispatchKeySet.h`, `aten/src/ATen/Dispatch.h`
3. `legacy_symbols`: `DispatchKey`, `DispatchKeySet`, `highestPriorityTypeId`, `highestPriorityBackendTypeId`, `AT_DISPATCH_SWITCH`
4. `tensor_storage_contract`: unchanged from `FT-P2C-001` scoped scalar storage
5. `dispatch_contract`: explicit keyset + fail-closed compatibility + mode-split fallback policy
6. `error_contract`: unknown key bits/incompatible keysets are explicit errors
7. `grad_graph_contract`: autograd path requests `AutogradCPU` when `requires_grad=true`
8. `serialization_contract`: no packet-specific change
9. `strict_mode_policy`: no composite/backend fallback
10. `hardened_mode_policy`: bounded fallback from composite/backend-select keys to backend key
11. `excluded_scope`: non-CPU backends, dynamic keys, boxed fallback
12. `oracle_tests`: `test/test_dispatch.py`, `test/test_python_dispatch.py` (scoped emulation)
13. `performance_sentinels`: dispatch key resolution latency + fallback rate
14. `compatibility_risks`: key precedence drift vs. PyTorch evolving enum
15. `raptorq_artifacts`: sidecar + decode proof emitted for packet parity report
