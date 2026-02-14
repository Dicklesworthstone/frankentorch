# EXECUTION_TODO_GRANULAR

Execution date: 2026-02-13  
Objective: complete `FT-P2C-002`, `FT-P2C-004`, and `FT-P2C-006` end-to-end (code, artifacts, conformance, optimization evidence, and validation) as part of a larger full-parity program.

## Absolute Parity Mandate
- This checklist captures a batch execution wave, not final scope.
- FrankenTorch release acceptance requires total feature/functionality overlap for drop-in replacement behavior.
- Any temporary sequencing gaps must be tracked as explicit parity-closure work with unit/property, differential/adversarial, and e2e-forensics evidence.

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

## 8. Current Bead Execution — bd-3v0.3
- [x] Run `bv --robot-triage` / `bv --robot-next` and select top actionable bead.
- [x] Mark `bd-3v0.3` as `in_progress`.
- [x] Reserve implementation files through Agent Mail.
- [x] Implement machine-checkable artifact schema lock validator.
- [x] Add schema lock spec doc (`SCHEMA_LOCK_V1.md`).
- [x] Run validator + required cargo gates after schema-lock change.
- [x] Send completion update via Agent Mail and mark bead status appropriately.

## 9. Current Bead Execution — bd-3v0.2
- [x] Select next top-impact bead via `bv --robot-next`.
- [x] Mark `bd-3v0.2` as `in_progress`.
- [x] Add versioned security/compatibility threat matrix.
- [x] Add hardened-mode deviation allowlist (explicit allowlist-only policy).
- [x] Extend schema validator with global checks for matrix+allowlist presence/shape.
- [x] Validate all current packets and global controls resolve `READY`.
- [x] Send completion updates through Agent Mail and close bead.

## 10. Extreme/Alien Uplift Pass (Code + Beads)

### 10.1 Optimization Loop — Validator Hotpath
- [x] Build isolated benchmark binary target (`/tmp/frankentorch-target`) to avoid cargo lock contention.
- [x] Create synthetic large-corpus benchmark root (`/tmp/ft_phase2c_bench_before_ep1sp0`) with 250 packet directories.
- [x] Record baseline wall-time with hyperfine (`mean 68.8 ms`).
- [x] Record baseline syscall profile with `strace -c` (`12,354` total calls).
- [x] Implement one optimization lever only: single-pass packet file cache in `validate_phase2c_artifacts`.
- [x] Re-run identical benchmark command and capture after metrics (`mean 64.7 ms`).
- [x] Re-run syscall profile and capture after metrics (`10,098` total calls).
- [x] Attach optimization evidence + risk note to bead `bd-3v0.8`.

### 10.2 Behavior-Isomorphism + Safety Gates
- [x] `cargo fmt --check`
- [x] `cargo check --all-targets`
- [x] `cargo clippy --all-targets -- -D warnings`
- [x] `cargo test --workspace`
- [x] `cargo test -p ft-conformance -- --nocapture`
- [x] `cargo bench`
- [x] Re-validate binary-scoped tests: `cargo test -p ft-conformance --bin validate_phase2c_artifacts`

### 10.3 Bead Graph Uplift (All Open/In-Progress `bd-3v0*`)
- [x] Export working set (`open + in_progress`) and quantify missing method-stack labels.
- [x] Add `extreme-software-optimization` label to every open/in-progress bead.
- [x] Add `alien-artifact-coding` label to every open/in-progress bead.
- [x] Add `alien-graveyard` label to every open/in-progress bead.
- [x] Verify zero missing labels for all three method tags.
- [x] Pull high-confidence (`>=0.95`) dependency suggestions via `bv --robot-suggest`.
- [x] Apply cycle-safe `missing_dependency` suggestions with `br dep add` (18 edges added).
- [x] Add explicit packet test-gating dependencies:
- [x] For `FT-P2C-001..008`, make every `*-G` (E2E/logging) depend on `*-E` (unit/property/logging).
- [x] For `FT-P2C-001..008`, make every `*-H` (optimization proof) depend on `*-E` (unit/property/logging).
- [x] For `FT-P2C-001..008`, make every `*-I` (final evidence) depend on both `*-E` and `*-F`.
- [x] Confirm added test-gating edges are cycle-safe (32 additional edges).
- [x] Reject cycle-inducing suggestions (2 rejected by cycle guard, intentionally not forced).
- [x] Verify dependency graph is acyclic (`br dep cycles --json` => empty).
- [x] Mark `bd-3v0.8` in-progress and log measured optimization evidence.
- [x] Post graph-uplift summary comment on master bead `bd-3v0`.

### 10.4 Multi-Agent Coordination
- [ ] Fetch/respond to Agent Mail inbox messages (blocked: MCP transport decode failure in this environment).
- [ ] Broadcast optimization + bead changes to agent peers after MCP transport recovery.

## 11. Structured Logging + E2E Matrix + Validator Parallelism (Current Pass)

### 11.1 Structured Logging Contract (bd-3v0.5 scope)
- [x] Add canonical structured test/e2e logging schema implementation (`ft-conformance-log-v1`).
- [x] Attach structured forensic logs to scalar/dispatch/scheduler/serialization case reports.
- [x] Enforce required log fields (`scenario_id`, `seed`, `mode`, `env_fingerprint`, `artifact_refs`, `replay_command`, `outcome`, `reason_code`).
- [x] Add contract document `artifacts/phase2c/TEST_LOG_CONTRACT_V1.md`.
- [x] Add unit tests proving log contract fields are populated.
- [x] Post implementation evidence comment on `bd-3v0.5` and close bead.

### 11.2 E2E Replay/Forensics Emitter
- [x] Add `emit_e2e_forensics_matrix` API to emit JSONL forensic logs.
- [x] Add CLI binary `run_e2e_matrix` with mode/output controls.
- [x] Add test validating JSONL emission and required key presence.
- [x] Add packet-scoped filtering (`--packet FT-P2C-00X`) and validation test.
- [x] Post implementation evidence comment on `bd-3v0.6` and close bead.

### 11.3 Extreme Optimization Lever (Retained)
- [x] Profile packet validator hotpath on synthetic large-corpus benchmark root.
- [x] Implement one lever only: packet-level parallel validation in `validate_phase2c_artifacts`.
- [x] Add deterministic fallback switch (`FT_DISABLE_PACKET_PARALLELISM=1`).
- [x] Benchmark baseline-vs-optimized in same revision and confirm speedup.
- [x] Record optimization + isomorphism artifact at `artifacts/optimization/2026-02-14_packet_parallel_validation.md`.

### 11.4 Optimization Candidate Rejected (Not Retained)
- [x] Evaluate typed fixture memoization in `ft-conformance`.
- [x] Re-benchmark and detect no win/regression on target workload.
- [x] Remove candidate implementation to avoid unnecessary complexity/perf drag.

### 11.5 Next Highest-Impact Bead Started
- [x] Re-run `bv --robot-next` and confirm next bead is `bd-3v0.4`.
- [x] Mark `bd-3v0.4` as `in_progress`.
- [x] Post progress comment describing closed gaps and remaining differential-oracle work.

## 12. Current Execution Wave (2026-02-14) — Differential Core + FT-P2C-001 Foundations

### 12.1 Absolute-Parity Doctrine Permeation
- [x] Harden top-level planning/spec docs to state total drop-in parity as non-negotiable.
- [x] Add `absolute-parity` + `drop-in-replacement` labels and explicit parity mandate block across active beads.
- [x] Verify no dependency cycles after doctrine updates.

### 12.2 Differential Conformance Harness Closure (`bd-3v0.4`)
- [x] Add legacy oracle execution path using real PyTorch (`.venv-py314`) for scoped packet families.
- [x] Add deterministic differential report schema with drift taxonomy.
- [x] Add hardened allowlist evaluation for policy deviations.
- [x] Add CLI emitter `run_differential_report`.
- [x] Emit `artifacts/phase2c/conformance/differential_report_v1.json`.
- [x] Add tests for differential sorting, JSON emission, and allowlist integrity.
- [x] Run full validation suite and close `bd-3v0.4`.

### 12.3 FT-P2C-001 Core Implementation Pass (`bd-3v0.12` in progress)
- [x] Strengthen `ft-core` metadata invariants: rank/stride checks, overflow-safe offset math, index bounds/rank guards.
- [x] Add explicit storage identity (`storage_id`) and alias-view semantics.
- [x] Add deterministic tensor/meta evidence fingerprints.
- [x] Add in-place version bump path and out-of-place version/storage behavior tests.
- [x] Expand `ft-core` unit suite to 10 tests with new invariant coverage.
- [x] Re-run workspace gates after core changes.
- [ ] Close packet contract/threat-model prerequisite beads (`bd-3v0.12.2`, `bd-3v0.12.3`) so child implementation beads can be claimed.

### 12.4 Essence Extraction Ledger (`bd-3v0.1`)
- [x] Create `artifacts/phase2c/ESSENCE_EXTRACTION_LEDGER_V1.md` with row-level anchors, uncertainty tags, and test/e2e traceability.
- [x] Link ledger from `PHASE2C_EXTRACTION_PACKET.md`.
- [x] Validate packet artifact integrity via `validate_phase2c_artifacts`.
- [x] Close `bd-3v0.1`.

### 12.5 Coordination + Residuals
- [x] Keep active bead status/comments synchronized via `br`.
- [x] Record Agent Mail transport unavailability and fallback coordination approach in bead comments.
- [ ] Resume direct agent-mail inbox/reply workflow once MCP transport is available.
