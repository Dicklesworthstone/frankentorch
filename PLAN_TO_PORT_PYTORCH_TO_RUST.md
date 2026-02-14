# PLAN_TO_PORT_PYTORCH_TO_RUST

## 1. Porting Methodology (Mandatory)

This project follows the spec-first `porting-to-rust` method:

1. Extract legacy behavior into executable specs.
2. Implement from spec (never line-by-line translation).
3. Prove parity via differential conformance.
4. Gate all optimization behind behavior-isomorphism proofs.
5. Maintain explicit closure paths until complete drop-in parity is achieved.

## 2. Legacy Oracle

- Path: `/dp/frankentorch/legacy_pytorch_code/pytorch`

## 3. Parity-Complete Target Surface

- Tensor metadata/storage/view/index semantics.
- Dispatch routing across full supported operator families.
- Autograd engine semantics and gradient correctness.
- Checkpoint/state serialization compatibility.
- NN + optimizer behavioral parity.
- Device/backend transition parity and execution-mode compatibility.

## 4. Sequencing Boundaries (No Permanent Exclusions)

- Execution is staged by packets to reduce risk and enforce proof quality.
- Deferred families remain mandatory parity-closure work, not optional scope.
- Every deferred behavior must have explicit downstream beads and conformance plans.

## 5. Phase Plan with Status

### Phase 1: Bootstrap + Planning (`complete`)
- parity doctrine and sequencing policy frozen
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

1. Differential parity green for full drop-in target surface.
2. No unresolved critical semantic drift.
3. Performance gates pass without correctness regressions.
4. RaptorQ sidecar artifacts validated for conformance + benchmark evidence.
5. No intentional feature/functionality omissions remain.
