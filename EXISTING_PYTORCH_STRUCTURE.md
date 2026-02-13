# EXISTING_PYTORCH_STRUCTURE

## 1. Legacy Oracle

- Root: `/dp/frankentorch/legacy_pytorch_code/pytorch`
- Upstream: `pytorch/pytorch`

## 2. Subsystem Map

- `c10/core`: tensor runtime primitives (`TensorImpl`, `TensorOptions`, `DispatchKey`, `Storage`, allocators).
- `aten/src/ATen/native`: eager kernels and native op registry metadata.
- `torch/csrc/autograd`: backward engine, GraphTask scheduling, hook points.
- `torch/csrc`: Python/C++ bridge and serialization surfaces.
- `torch/csrc/{cuda,mps,xpu}` + `c10` backend folders: device runtime glue.

## 3. Semantic Hotspots (Must Preserve)

1. TensorImpl/storage ownership and metadata invariants.
2. Dispatch key ordering and highest-priority kernel resolution.
3. Autograd engine semantics:
   - reentrant backward behavior
   - ready queue scheduling
   - stream synchronization contracts
4. SavedVariable + hook lifetime semantics.
5. Gradient mode and dispatch layer interactions at Python bridge boundaries.

## 4. Compatibility-Critical Behaviors

- Dispatcher fallback/table lookup behavior.
- Backward ordering and stream contracts.
- Checkpoint/serialization format behavior for scoped state.
- Device guard behavior in mixed backend execution paths.

## 5. Security and Stability Risk Areas

- serialization and pickle-adjacent surfaces.
- distributed/RPC tensor messaging and lifecycle assumptions.
- GPU stream/event lifecycle correctness.
- autograd race/deadlock hazards under worker scheduling.

## 6. V1 Extraction Boundary

### Include now

- `c10` runtime primitives.
- eager CPU kernel subset.
- autograd engine core.
- essential bridge glue for conformance.

### Exclude for V1

- TorchScript/JIT breadth.
- distributed/RPC breadth.
- compiler-stack breadth.
- mobile wrappers.

## 7. High-Value Conformance Fixture Families

- c10 dispatch key tests.
- ATen tensor/kernel subsets.
- autograd fixture families in `torch/testing/_internal`.
- serialization and state round-trip subsets.

## 8. First Packet Anchors Already Extracted

See `artifacts/phase2c/FT-P2C-001/legacy_anchor_map.md` for exact `TensorImpl.h` anchors covering:
- `AutogradMetaInterface`
- `VariableVersion`
- metadata fields (`sizes_and_strides_`, `storage_offset_`, `numel_`, `data_type_`, `device_opt_`)

## 9. Extraction Notes for Rust Spec

- Freeze dispatch key model before broad kernel expansion.
- Build gradient parity checks early to prevent semantic drift.
- Treat serialization as a compatibility contract, not an implementation detail.
