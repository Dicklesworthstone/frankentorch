# COMPREHENSIVE_SPEC_FOR_FRANKENTORCH_V1

Document-version note: "V1" is spec revisioning, not a reduced parity target. This spec governs the full drop-in replacement objective.

## 0. Prime Directive

Build a system that is simultaneously:

1. Behaviorally trustworthy for full drop-in compatibility.
2. Mathematically explicit in decision and risk handling.
3. Operationally resilient via RaptorQ-backed durability.
4. Performance-competitive via profile-and-proof discipline.

Crown-jewel innovation:

Deterministic Autograd Contract (DAC): replayable gradient graph execution with provenance-complete gradient evidence.

Legacy oracle:

- /dp/frankentorch/legacy_pytorch_code/pytorch
- upstream: https://github.com/pytorch/pytorch

## 1. Product Thesis

Most reimplementations fail by being partially compatible and operationally brittle. FrankenTorch will instead combine compatibility realism with first-principles architecture and strict quality gates.

## 2. Absolute Parity Contract

FrankenTorch is specified as a full drop-in replacement target for upstream PyTorch observable behavior.

Execution is packetized and staged, but staging is sequencing only, never a release-time scope reduction.

Temporary sequencing policy:
- deferred work must be represented as explicit parity-closure beads with dependencies
- every deferred behavior must include oracle fixture and conformance plans
- release sign-off requires zero intentional feature/functionality gaps

## 3. Architecture Blueprint

tensor API -> dispatcher -> device kernels -> autograd engine -> nn and optimizer stack

Planned crate families:
- ft-types
- ft-tensor
- ft-dispatch
- ft-autograd
- ft-ops
- ft-nn
- ft-optim
- ft-serialize
- ft-conformance
- frankentorch

## 4. Compatibility Model (frankenlibc/frankenfs-inspired)

Two explicit operating modes:

1. strict mode:
   - maximize observable compatibility for full drop-in behavior
   - no behavior-altering repair heuristics
2. hardened mode:
   - maintain outward contract while enabling defensive runtime checks and bounded repairs

Compatibility focus for this project:

Preserve full PyTorch-observable tensor semantics, autograd contracts, optimizer/module behaviors, serialization contracts, and execution invariants required for drop-in use.

Fail-closed policy:

- unknown incompatible features or protocol fields must fail closed by default
- compatibility exceptions require explicit allowlist entries and audit traces

## 5. Security Model

Security focus for this project:

Protect against gradient corruption, unsafe in-place mutation paths, and serialization mismatch or replay inconsistencies.

Threat model baseline:

1. malformed input and parser abuse
2. state-machine desynchronization
3. downgrade and compatibility confusion paths
4. persistence corruption and replay tampering

Mandatory controls:

- adversarial fixtures and fuzz/property suites for high-risk entry points
- deterministic audit trail for recoveries and mode/policy overrides
- explicit subsystem ownership and trust-boundary notes

## 6. Alien-Artifact Decision Layer

Runtime controllers (scheduling, adaptation, fallback, admission) must document:

1. state space
2. evidence signals
3. loss matrix with asymmetric costs
4. posterior or confidence update model
5. action rule minimizing expected loss
6. calibration fallback trigger

Output requirements:

- evidence ledger entries for consequential decisions
- calibrated confidence metrics and drift alarms

## 7. Extreme Optimization Contract

Track step-time tails, backward overhead, kernel throughput, and memory churn under representative training traces.

Optimization loop is mandatory:

1. baseline metrics
2. hotspot profile
3. single-lever optimization
4. behavior-isomorphism proof
5. re-profile and compare

No optimization is accepted without associated correctness evidence.

## 8. Correctness and Conformance Contract

Maintain deterministic gradient accumulation, alias/versioning rules, and backward-equivalence invariants.

Conformance process:

1. generate canonical fixture corpus
2. run legacy oracle and capture normalized outputs
3. run FrankenTorch and compare under explicit equality/tolerance policy
4. produce machine-readable parity report artifact

Assurance ladder:

- Tier A: unit/integration/golden fixtures
- Tier B: differential conformance
- Tier C: property/fuzz/adversarial tests
- Tier D: regression corpus for historical failures

## 9. RaptorQ-Everywhere Durability Contract

RaptorQ repair-symbol sidecars are required for long-lived project evidence:

1. conformance snapshots
2. benchmark baselines
3. migration manifests
4. reproducibility ledgers
5. release-grade state artifacts

Required artifacts:

- symbol generation manifest
- scrub verification report
- decode proof for each recovery event

## 10. Milestones and Exit Criteria

### M0 — Bootstrap

- workspace skeleton
- CI and quality gate wiring

Exit:
- fmt/check/clippy/test baseline green

### M1 — Core Model

- core data/runtime structures
- first invariant suite

Exit:
- invariant suite green
- first conformance fixtures passing

### M2 — First Vertical Slice

- end-to-end workflow implemented with explicit parity-closure backlog

Exit:
- differential parity for first major API family
- baseline benchmark report published

### M3 — Parity Expansion

- additional API families closed toward total parity

Exit:
- expanded parity reports green
- no unresolved critical compatibility defects

### M4 — Hardening

- adversarial coverage and perf hardening

Exit:
- regression gates stable
- conformance drift zero for full drop-in target surface

## 11. Acceptance Gates

Gate A: compatibility parity report passes for full drop-in target surface.

Gate B: security/fuzz/adversarial suite passes for high-risk paths.

Gate C: performance budgets pass with no semantic regressions.

Gate D: RaptorQ durability artifacts validated and scrub-clean.

All four gates must pass for release readiness.

## 12. Risk Register

Primary risk focus:

Autograd drift introduced by dispatch/kernel optimization without full replay proofs.

Mitigations:

1. compatibility-first development for risky API families
2. explicit invariants and adversarial tests
3. profile-driven optimization with proof artifacts
4. strict mode/hardened mode separation with audited policy transitions
5. RaptorQ-backed resilience for critical persistent artifacts

## 13. Immediate Execution Checklist

1. Create workspace and crate skeleton.
2. Implement smallest high-value end-to-end path and maintain explicit parity-closure chain to full coverage.
3. Stand up differential conformance harness against legacy oracle.
4. Add benchmark baseline generation and regression gating.
5. Add RaptorQ sidecar pipeline for conformance and benchmark artifacts.

## 14. Detailed Crate Contracts

| Crate | Primary Responsibility | Explicit Non-Goal | Invariants | Mandatory Tests |
|---|---|---|---|---|
| ft-types | dtype/device/layout metadata model | autograd execution | metadata validity and versioning | metadata fixture matrix |
| ft-tensor | tensor storage/view/index semantics | optimizer logic | alias/version consistency and shape safety | view/in-place property tests |
| ft-dispatch | op registration and dispatch key routing | model serialization | deterministic dispatch resolution | dispatch key regression suite |
| ft-autograd | graph capture + backward execution | parser/IO | gradient accumulation and graph ordering contracts | gradient check suite |
| ft-ops | core kernels | policy/risk logic | kernel outputs satisfy tolerance contracts | kernel parity + tolerance tests |
| ft-nn | module stack | backend kernels | parameter and module state determinism | module fixture tests |
| ft-optim | optimizer updates | tensor storage internals | update equations and state transitions deterministic | SGD/Adam parity tests |
| ft-serialize | checkpoints/state dict paths | training execution | serialization round-trip parity | checkpoint fixtures |
| ft-conformance | PyTorch differential harness | production dispatch | comparator policy explicit by op/dtype | differential harness tests |
| frankentorch | integration + mode policy | low-level kernels | strict/hardened mode and evidence ledger available | integration startup tests |

## 15. Conformance Matrix

| Family | Oracle Workload | Pass Criterion | Drift Severity |
|---|---|---|---|
| Tensor creation/view | shape/stride fixture set | exact metadata parity | critical |
| Elementwise ops | arithmetic op corpus | value parity under tolerance policy | high |
| Matmul/reduction | core linear algebra suite | value and shape parity | critical |
| Autograd backward | gradient-check corpus | gradient parity within budget | critical |
| In-place behavior | mutation/version fixtures | alias/version parity | critical |
| nn module forward | baseline model suite with closure expansion | output parity | high |
| Optimizer step | SGD/Adam traces | parameter update parity | critical |
| Checkpoint round-trip | save/load suites | state parity | high |

## 16. Security and Compatibility Threat Matrix

| Threat | Strict Mode Response | Hardened Mode Response | Required Artifact |
|---|---|---|---|
| Malformed checkpoint payload | fail-closed | fail-closed + bounded diagnostics | serialization incident ledger |
| Gradient poisoning via invalid graph | fail invalid graph | quarantine and fail with trace | autograd validation report |
| Dispatch confusion attack | fail ambiguous dispatch | fail ambiguous dispatch | dispatch audit report |
| In-place mutation misuse | strict version checks | stricter runtime guards | version safety ledger |
| Unknown incompatible checkpoint version | fail-closed | fail-closed | compatibility drift report |
| Differential mismatch | release gate failure | release gate failure | conformance failure bundle |
| Corrupt artifacts | reject | recover via RaptorQ + verify | decode proof + scrub report |
| Unauthorized policy overrides | audited override only | audited override only | override audit log |

## 17. Performance Budgets and SLO Targets

| Path | Workload | Budget |
|---|---|---|
| elementwise forward | 100M elements | p95 <= 200 ms |
| backward pass | representative autograd graphs | p95 <= 1.35x forward baseline |
| matmul core | medium dense matrices | p95 <= 240 ms |
| optimizer step | 10M params Adam | p95 <= 130 ms |
| checkpoint save/load | 2 GB model state | throughput >= 350 MB/s |
| memory footprint | training step profile | peak RSS regression <= +8% |
| allocation churn | backward-heavy trace | alloc count regression <= +10% |
| tail stability | benchmark families | p99 regression <= +7% |

## 18. CI Gate Topology (Release-Critical)

| Gate | Name | Blocking | Output Artifact |
|---|---|---|---|
| G1 | format + lint | yes | lint report |
| G2 | unit + integration | yes | junit report |
| G3 | differential + gradient checks | yes | parity report + grad report |
| G4 | adversarial + property tests | yes | counterexample corpus |
| G5 | benchmark regression | yes | baseline delta report |
| G6 | RaptorQ scrub + recovery drill | yes | scrub report + decode proof |

## 19. RaptorQ Artifact Envelope (Project-Wide)

~~~json
{
  "artifact_id": "string",
  "artifact_type": "conformance|benchmark|ledger|manifest",
  "source_hash": "blake3:...",
  "raptorq": {
    "k": 0,
    "repair_symbols": 0,
    "overhead_ratio": 0.0
  },
  "scrub": {
    "status": "ok|recovered|failed",
    "last_ok_unix_ms": 0
  },
  "decode_proofs": [
    {
      "ts_unix_ms": 0,
      "reason": "...",
      "proof_hash": "blake3:..."
    }
  ]
}
~~~

## 20. 90-Day Execution Plan

Weeks 1-2:
- scaffold crate boundaries and tensor metadata contracts

Weeks 3-5:
- tensor core + autograd baseline path with explicit closure chain
- strict conformance and gradient-check harness

Weeks 6-8:
- nn/optimizer workflows + baseline benchmarks

Weeks 9-10:
- checkpoint/parser hardening and adversarial fixtures
- strict/hardened mode policy + audited overrides

Weeks 11-12:
- enforce full G1-G6 topology and run release-candidate drill

## 21. Porting Artifact Index

This spec is paired with the following methodology artifacts:

1. PLAN_TO_PORT_PYTORCH_TO_RUST.md
2. EXISTING_PYTORCH_STRUCTURE.md
3. PROPOSED_ARCHITECTURE.md
4. FEATURE_PARITY.md

Rule of use:

- Extraction and behavior understanding happens in EXISTING_PYTORCH_STRUCTURE.md.
- Sequencing and parity-closure dependency policy live in PLAN_TO_PORT_PYTORCH_TO_RUST.md.
- Rust crate boundaries live in PROPOSED_ARCHITECTURE.md.
- Delivery readiness is tracked in FEATURE_PARITY.md.

## 22. FrankenSQLite Exemplar Lock (Normative)

FrankenTorch SHALL treat `COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1_REFERENCE.md` as a normative methodology exemplar for:

1. section-level precision and explicit invariants,
2. strict/hardened mode governance,
3. evidence-ledger-first operability,
4. RaptorQ durability envelopes and decode-proof chains,
5. release-gating conformance discipline.

Any major FrankenTorch spec evolution SHOULD include an explicit crosswalk entry in:
- `FRANKENSQLITE_ADAPTATION_CROSSWALK.md`

## 23. Packetized Porting Governance (Mandatory)

All implementation work SHALL map to `FT-P2C-*` packets with artifacts at:
- `artifacts/phase2c/FT-P2C-00X/legacy_anchor_map.md`
- `artifacts/phase2c/FT-P2C-00X/contract_table.md`
- `artifacts/phase2c/FT-P2C-00X/fixture_manifest.json`
- `artifacts/phase2c/FT-P2C-00X/parity_gate.yaml`
- `artifacts/phase2c/FT-P2C-00X/risk_note.md`

Packet completion requires strict+hardened conformance green for the packet fixture family.

## 24. First Vertical Slice Record (Delivered)

Delivered in this session:

- scalar tensor metadata/value/version kernel path
- deterministic scalar autograd replay with gradient trace
- strict+hardened fixture-driven scalar conformance
- runtime evidence ledger and durability envelope model

Implementation anchors:
- `crates/ft-core/src/lib.rs`
- `crates/ft-dispatch/src/lib.rs`
- `crates/ft-kernel-cpu/src/lib.rs`
- `crates/ft-autograd/src/lib.rs`
- `crates/ft-runtime/src/lib.rs`
- `crates/ft-api/src/lib.rs`
- `crates/ft-conformance/src/lib.rs`


## 27. Phase-2C Delta (2026-02-13)

Implemented packets in this revision:

1. `FT-P2C-002`:
   - `DispatchKey` + `DispatchKeySet` contract with explicit priority ordering.
   - strict fail-closed behavior for incompatible composite/backend fallback routes.
   - hardened bounded fallback with explicit evidence metadata.
2. `FT-P2C-004`:
   - deterministic dependency-driven ready-queue scheduler.
   - explicit reentrant-depth policy split (`strict` fail vs. `hardened` bounded fallback).
   - scheduler telemetry promoted to first-class backward evidence.
3. `FT-P2C-006`:
   - typed checkpoint envelope with `deny_unknown_fields` strict parsing.
   - deterministic checksum and version gates.
   - RaptorQ sidecar + decode proof generation through `asupersync`.

Artifact topology for these packets is populated under `artifacts/phase2c/FT-P2C-002`, `artifacts/phase2c/FT-P2C-004`, and `artifacts/phase2c/FT-P2C-006`.
