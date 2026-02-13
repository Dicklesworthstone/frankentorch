# PLAN_TO_PORT_PYTORCH_TO_RUST

## 1. Porting Methodology (Mandatory)

This project follows the spec-first `porting-to-rust` method:

1. Extract legacy behavior into executable specs.
2. Implement from spec (never line-by-line translation).
3. Prove parity via differential conformance.
4. Gate all optimization behind behavior-isomorphism proofs.

## 2. Legacy Oracle

- Path: `/dp/frankentorch/legacy_pytorch_code/pytorch`

## 3. V1 Scope

- Tensor metadata/storage/view/index semantics.
- Dispatch routing for scoped op families.
- Autograd engine core semantics and gradient correctness.
- Scoped checkpoint/state serialization compatibility.
- Minimal NN + optimizer first-wave behavior.

## 4. Explicit Exclusions for V1

- TorchScript/JIT and distributed RPC breadth.
- Compiler-stack breadth (`dynamo`, `inductor`, `functorch`) in V1.
- Mobile/platform wrappers and broad ecosystem tooling.

## 5. Phase Plan with Status

### Phase 1: Bootstrap + Planning (`complete`)
- scope and exclusions frozen
- compatibility contract drafted

### Phase 2: Deep Structure Extraction (`in_progress`)
- `EXISTING_PYTORCH_STRUCTURE.md` expanded
- packetized extraction program (`FT-P2C-*`) established

### Phase 3: Architecture Synthesis (`in_progress`)
- crate boundaries and mode-split policy documented
- frankensqlite adaptation crosswalk added

### Phase 4: Implementation (`in_progress`)
- first deterministic scalar DAC vertical slice shipped:
  - `ft-core`, `ft-dispatch`, `ft-kernel-cpu`, `ft-autograd`, `ft-runtime`, `ft-api`

### Phase 5: Conformance and QA (`in_progress`)
- fixture-driven strict+hardened scalar conformance green
- benchmark harness entrypoint added (`run_scalar_microbench`)

## 6. Mandatory Exit Criteria

1. Differential parity green for scoped APIs.
2. No unresolved critical semantic drift.
3. Performance gates pass without correctness regressions.
4. RaptorQ sidecar artifacts validated for conformance + benchmark evidence.
