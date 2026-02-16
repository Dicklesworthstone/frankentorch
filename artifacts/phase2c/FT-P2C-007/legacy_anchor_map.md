# FT-P2C-007 — Legacy Anchor Map

Packet: Device guard and backend transitions  
Legacy root: `legacy_pytorch_code/pytorch`

## Extracted Anchors (Exact)

| Legacy path | Line anchor | Symbol | Porting relevance |
|---|---:|---|---|
| `c10/core/DeviceGuard.h` | 23 | `class DeviceGuard` | RAII device-guard contract for scoped backend/device transitions |
| `c10/core/DeviceGuard.h` | 55 | `void reset_device(at::Device device)` | explicit device transition primitive and consistency boundary |
| `c10/core/DeviceGuard.h` | 130 | `class OptionalDeviceGuard` | optional guard lifecycle used when device context may be absent |
| `aten/src/ATen/core/op_registration/adaption.h` | 47 | `check_and_update_common_device(...)` | canonical same-device aggregation and mismatch detection |
| `aten/src/ATen/core/adaption.cpp` | 6 | `common_device_check_failure(...)` | standardized fail-closed device mismatch diagnostic text |
| `aten/src/ATen/TensorIterator.cpp` | 492 | `config.check_all_same_device_` branch | tensor-iterator level same-device enforcement |
| `aten/src/ATen/TensorIterator.cpp` | 502 | `"Expected all tensors to be on the same device"` | user-observable mismatch error wording baseline |
| `c10/core/DispatchKey.h` | 219 | `BackendSelect` | backend resolution key for tensor-less or ambiguous routing contexts |
| `c10/core/DispatchKey.h` | 459 | `CompositeImplicitAutograd` | alias-key fallback tier in dispatch table resolution |
| `c10/core/DispatchKey.h` | 472 | `CompositeExplicitAutograd` | explicit composite fallback tier above implicit autograd |
| `c10/core/DispatchKeySet.h` | 674 | `default_included_set` includes `BackendSelect` | baseline TLS-included dispatch behavior |
| `c10/core/DispatchKeySet.h` | 761 | `autograd_cpu_ks` | explicit AutogradCPU keyset representation |
| `c10/core/Backend.h` | 74 | `dispatchKeyToBackend(AutogradCPU) -> CPU` | backend canonicalization for autograd CPU key |
| `aten/src/ATen/core/dispatch/OperatorEntry.cpp` | 352 | `computeDispatchTableEntryWithDebug(...)` | runtime key resolution ordering contract |
| `aten/src/ATen/core/dispatch/OperatorEntry.cpp` | 358 | alias precedence block (`Composite*`, `Autograd`) | strict ordering of composite/autograd fallback tiers |
| `aten/src/ATen/core/dispatch/OperatorEntry.cpp` | 460 | backend fallback branch | final fallback stage before missing-kernel error |

## Behavioral Oracle Tests (Scoped)

| Legacy path | Intent |
|---|---|
| `test/test_binary_ufuncs.py` (`1186`, `1486`, `1691`, `2316`, `2321`, `3378`, `3383`) | cross-device and mismatch-path assertions for binary ops |
| `test/test_torch.py` (`4395`, `4406`, `5517`) | broad same-device runtime error contract checks |
| `c10/test/core/DeviceGuard_test.cpp` (`14`, `29`) | guard reset semantics across device-type transitions |

## Implemented Rust Mapping

| Rust crate | File | Mapping |
|---|---|---|
| `ft-device` | `crates/ft-device/src/lib.rs` | `DeviceGuard` + `ensure_same_device` fail-closed device contract |
| `ft-dispatch` | `crates/ft-dispatch/src/lib.rs` | dispatch-keyset validation, composite mode-split policy, backend-resolution envelope |
| `ft-conformance` | `crates/ft-conformance/fixtures/dispatch_key_cases.json` | strict/hardened transition fixture families for composite/autograd/device mismatch paths |
| `ft-conformance` | `crates/ft-conformance/src/lib.rs` | dispatch conformance harness and differential policy checks for mode-split/fail-closed behavior |

## Extraction Schema (Mandatory)

1. `packet_id`: `FT-P2C-007`
2. `legacy_paths`: `c10/core/DeviceGuard.h`, `aten/src/ATen/core/op_registration/adaption.h`, `aten/src/ATen/core/adaption.cpp`, `aten/src/ATen/TensorIterator.cpp`, `c10/core/DispatchKey.h`, `c10/core/DispatchKeySet.h`, `c10/core/Backend.h`, `aten/src/ATen/core/dispatch/OperatorEntry.cpp`
3. `legacy_symbols`: `DeviceGuard`, `OptionalDeviceGuard`, `check_and_update_common_device`, `common_device_check_failure`, `BackendSelect`, `CompositeExplicitAutograd`, `CompositeImplicitAutograd`, `computeDispatchTableEntryWithDebug`
4. `tensor_storage_contract`: binary-op routes must reject cross-device tensor pairs unless explicitly supported by contract
5. `dispatch_contract`: keyset validation + deterministic type/backend resolution; strict mode forbids composite/backend fallback transitions
6. `error_contract`: unknown/empty/incompatible keysets and dtype/device mismatches fail closed with stable diagnostics
7. `grad_graph_contract`: AutogradCPU route remains valid only when CPU backend availability is present
8. `serialization_contract`: no packet-local serialization format change
9. `strict_mode_policy`: reject composite/backend fallback routes and malformed keysets with zero recovery
10. `hardened_mode_policy`: permit bounded composite/backend fallback only when backend key resolves safely
11. `excluded_scope`: non-CPU backend expansion (CUDA/XPU/MPS/etc.), heterogeneous stream/device-guard implementations beyond current CPU-focused packet
12. `oracle_tests`: `test/test_binary_ufuncs.py`, `test/test_torch.py`, `c10/test/core/DeviceGuard_test.cpp`
13. `performance_sentinels`: dispatch-keyset validation overhead, fallback-branch frequency, device-guard check costs under representative traces
14. `compatibility_risks`: backend key domain remains intentionally narrow vs full upstream backend matrix; expansion deferred to downstream packet work
15. `raptorq_artifacts`: packet parity sidecar + decode-proof closure remain mandatory in final evidence bead (`bd-3v0.18.9`)
