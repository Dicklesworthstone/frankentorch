# EXECUTION_TODO_GRANULAR

Execution date: 2026-02-13  
Objective: complete `FT-P2C-002`, `FT-P2C-004`, and `FT-P2C-006` end-to-end (code, artifacts, conformance, optimization evidence, and validation).

## 0. Governance + Tracking
- [x] Create persistent granular TODO list file.
- [x] Keep this checklist updated after each completed subtask.
- [x] Ensure no destructive operations are executed.

## 1. FT-P2C-002 — Dispatch Key Model

### 1.1 Spec Extraction + Anchors
- [x] Extract exact anchors from `c10/core/DispatchKey.h`.
- [x] Extract exact anchors from `c10/core/DispatchKeySet.h`.
- [x] Extract exact anchors from `aten/src/ATen/Dispatch.h`.
- [x] Record extracted symbols and behavior in packet `legacy_anchor_map.md`.

### 1.2 Implementation
- [x] Add `DispatchKey` model in `ft-dispatch` with explicit precedence.
- [x] Add `DispatchKeySet` bitset model with `add/remove/has/union/intersection`.
- [x] Add highest-priority resolution logic (`highest_priority_type_id`).
- [x] Add backend-priority helper (`highest_priority_backend_type_id`).
- [x] Add fail-closed handling for unknown/incompatible keysets.
- [x] Update `DispatchDecision` to carry selected key and keyset representation.
- [x] Update scalar dispatch path to use keyset resolution.

### 1.3 Tests + Conformance
- [x] Add unit tests for keyset operations.
- [x] Add unit tests for priority resolution ordering.
- [x] Add unit tests for fail-closed unknown/incompatible cases.
- [x] Add fixture family for dispatch key conformance.
- [x] Add `ft-conformance` checks for strict+hardened dispatch behavior.

### 1.4 Packet Artifacts
- [x] Create `artifacts/phase2c/FT-P2C-002/legacy_anchor_map.md`.
- [x] Create `artifacts/phase2c/FT-P2C-002/contract_table.md`.
- [x] Create `artifacts/phase2c/FT-P2C-002/fixture_manifest.json`.
- [x] Create `artifacts/phase2c/FT-P2C-002/parity_gate.yaml`.
- [x] Create `artifacts/phase2c/FT-P2C-002/risk_note.md`.
- [x] Create `artifacts/phase2c/FT-P2C-002/parity_report.json`.
- [x] Create `artifacts/phase2c/FT-P2C-002/parity_report.raptorq.json`.
- [x] Create `artifacts/phase2c/FT-P2C-002/parity_report.decode_proof.json`.

## 2. FT-P2C-004 — Autograd Engine Scheduling

### 2.1 Spec Extraction + Anchors
- [x] Extract exact anchors from `torch/csrc/autograd/engine.h` (`NodeTask`, `ReadyQueue`, `Engine::execute`).
- [x] Extract exact anchors from `torch/csrc/autograd/engine.cpp` (`thread_main`, `compute_dependencies`, reentrant depth).
- [x] Record extracted symbols and behavior in packet `legacy_anchor_map.md`.

### 2.2 Implementation
- [x] Implement deterministic ready-queue scheduler model in `ft-autograd`.
- [x] Implement dependency counting (`compute_dependencies` analog).
- [x] Implement priority ordering policy for queued tasks.
- [x] Implement explicit reentrant depth options and max-depth guard.
- [x] Implement strict-mode reentrant fail behavior.
- [x] Implement hardened-mode bounded fallback behavior.
- [x] Expose scheduler telemetry in `BackwardReport`.
- [x] Update `ft-api` session backward path to pass mode-aware options.

### 2.3 Tests + Conformance
- [x] Add scheduler ordering tests.
- [x] Add dependency completion tests.
- [x] Add reentrant-depth strict fail test.
- [x] Add reentrant-depth hardened fallback test.
- [x] Add autograd scheduling fixture family in `ft-conformance`.
- [x] Add strict+hardened autograd scheduling conformance checks.

### 2.4 Packet Artifacts
- [x] Create `artifacts/phase2c/FT-P2C-004/legacy_anchor_map.md`.
- [x] Create `artifacts/phase2c/FT-P2C-004/contract_table.md`.
- [x] Create `artifacts/phase2c/FT-P2C-004/fixture_manifest.json`.
- [x] Create `artifacts/phase2c/FT-P2C-004/parity_gate.yaml`.
- [x] Create `artifacts/phase2c/FT-P2C-004/risk_note.md`.
- [x] Create `artifacts/phase2c/FT-P2C-004/parity_report.json`.
- [x] Create `artifacts/phase2c/FT-P2C-004/parity_report.raptorq.json`.
- [x] Create `artifacts/phase2c/FT-P2C-004/parity_report.decode_proof.json`.

## 3. FT-P2C-006 — Serialization + RaptorQ Sidecar

### 3.1 Spec Extraction + Anchors
- [x] Extract exact anchors for `THPStorage_writeFileRaw` from `torch/csrc/serialization.cpp`.
- [x] Extract exact anchors for `THPStorage_readFileRaw` from `torch/csrc/serialization.cpp`.
- [x] Extract compatibility-relevant behavior notes (endianness, exact size checks, EOF fail behavior).
- [x] Record extracted symbols and behavior in packet `legacy_anchor_map.md`.

### 3.2 Implementation
- [x] Add typed checkpoint schema in `ft-serialize`.
- [x] Add strict decode path with fail-closed unknown fields.
- [x] Add hardened decode path with bounded diagnostics (while preserving fail-closed for incompatible fields).
- [x] Add explicit version gate behavior.
- [x] Add deterministic checksum/hash field generation for checkpoint payload.
- [x] Integrate `asupersync` RaptorQ encode/decode proof flow.
- [x] Implement sidecar generation manifest with repair symbol metadata.
- [x] Implement decode proof capture and content hash persistence.

### 3.3 Tests + Conformance
- [x] Add round-trip serialization tests.
- [x] Add strict unknown-field fail tests.
- [x] Add hardened malformed payload diagnostic tests.
- [x] Add sidecar generation tests.
- [x] Add decode proof determinism tests.
- [x] Add serialization fixture family in `ft-conformance`.
- [x] Add strict+hardened serialization conformance checks.

### 3.4 Packet Artifacts
- [x] Create `artifacts/phase2c/FT-P2C-006/legacy_anchor_map.md`.
- [x] Create `artifacts/phase2c/FT-P2C-006/contract_table.md`.
- [x] Create `artifacts/phase2c/FT-P2C-006/fixture_manifest.json`.
- [x] Create `artifacts/phase2c/FT-P2C-006/parity_gate.yaml`.
- [x] Create `artifacts/phase2c/FT-P2C-006/risk_note.md`.
- [x] Create `artifacts/phase2c/FT-P2C-006/parity_report.json`.
- [x] Create `artifacts/phase2c/FT-P2C-006/parity_report.raptorq.json`.
- [x] Create `artifacts/phase2c/FT-P2C-006/parity_report.decode_proof.json`.

## 4. Cross-Cutting Conformance + Optimization Evidence
- [x] Extend `ft-conformance` harness to run all packet fixture families.
- [x] Update smoke report summary to include dispatch/autograd/serialization packet status.
- [x] Refresh optimization opportunity matrix with new hotspots.
- [x] Add isomorphism proof blocks for each implemented lever.
- [x] Refresh golden outputs/checksums where behavior changed.

## 5. Documentation + Status Rollup
- [x] Update `FEATURE_PARITY.md` with packet-level progress.
- [x] Update `PHASE2C_EXTRACTION_PACKET.md` status section for 002/004/006.
- [x] Update relevant spec docs if new contracts were introduced.
- [x] Confirm method-stack artifact production vs deferral.

## 6. Validation Gates (Mandatory)
- [x] `cargo fmt --check`
- [x] `cargo check --all-targets`
- [x] `cargo clippy --all-targets -- -D warnings`
- [x] `cargo test --workspace`
- [x] `cargo test -p ft-conformance -- --nocapture`
- [x] `cargo bench`
- [x] verify checksum artifacts (`sha256sum -c ...`)

## 7. Finalization
- [x] Re-scan checklist for any unchecked tasks and complete or explicitly defer with rationale.
- [x] Summarize completed work, residual risks, and next-highest-value tasks.
- [x] Explicitly confirm no destructive operations were used.
