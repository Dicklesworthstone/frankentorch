# FrankenTorch

<div align="center">
  <img src="frankentorch_illustration.webp" alt="FrankenTorch - memory-safe clean-room Rust reimplementation of PyTorch">
</div>

FrankenTorch is a clean-room Rust reimplementation targeting grand-scope excellence: semantic fidelity, mathematical rigor, operational safety, and profile-proven performance.

## Absolute Parity Doctrine

FrankenTorch is not a partial rewrite. The target is a true drop-in replacement with complete feature and functionality overlap with upstream PyTorch observable behavior.

Non-negotiable rules:
- no permanent scope cuts accepted as a release condition
- no "minimal viable parity" acceptance standard
- sequencing is allowed only as temporary execution order, never as feature removal
- every temporary gap must map to explicit parity-closure beads plus conformance evidence

## What Makes This Project Special

Deterministic Autograd Contract (DAC): replayable gradient graph execution with provenance-complete gradient evidence.

This is treated as a core identity constraint, not a best-effort nice-to-have.

## Methodological DNA

This project uses four pervasive disciplines:

1. alien-artifact-coding for decision theory, confidence calibration, and explainability.
2. extreme-software-optimization for profile-first, proof-backed performance work.
3. RaptorQ-everywhere for self-healing durability of long-lived artifacts and state.
4. frankenlibc/frankenfs compatibility-security thinking: strict vs hardened mode separation, fail-closed compatibility gates, and explicit drift ledgers.

## Current State

- project charter docs established
- legacy oracle cloned:
  - /dp/frankentorch/legacy_pytorch_code/pytorch

## Delivery Sequencing (Not Scope Reduction)

Implementation proceeds in packetized waves (`FT-P2C-*`) to control risk and improve proof quality, but the terminal target remains complete PyTorch drop-in compatibility.

## Architecture Direction

tensor API -> dispatcher -> device kernels -> autograd engine -> nn and optimizer stack

## Compatibility and Security Stance

Preserve full PyTorch-observable tensor semantics, autograd contracts, optimizer behaviors, serialization contracts, and runtime compatibility invariants required for drop-in use.

Protect against gradient corruption, unsafe in-place mutation paths, and serialization mismatch or replay inconsistencies.

## Performance and Correctness Bar

Track step-time tails, backward overhead, kernel throughput, and memory churn under representative training traces.

Maintain deterministic gradient accumulation, alias/versioning rules, and backward-equivalence invariants.

## Key Documents

- AGENTS.md
- COMPREHENSIVE_SPEC_FOR_FRANKENTORCH_V1.md

## Conformance + E2E Forensics

- Run packet-side conformance:
  - `cargo test -p ft-conformance -- --nocapture`
- Emit structured e2e forensic JSONL logs:
  - `cargo run -p ft-conformance --bin run_e2e_matrix -- --mode both --output artifacts/phase2c/e2e_forensics/e2e_matrix.jsonl`
- Packet-scoped e2e forensic slice:
  - `cargo run -p ft-conformance --bin run_e2e_matrix -- --mode both --packet FT-P2C-004 --output artifacts/phase2c/e2e_forensics/ft-p2c-004.jsonl`
- Log schema contract:
  - `artifacts/phase2c/TEST_LOG_CONTRACT_V1.md`

## Next Steps

1. Scaffold Cargo workspace and crate map.
2. Expand from the first shipped vertical slice through parity-closure packets until all intentional parity gaps are closed.
3. Establish differential conformance harness vs legacy oracle.
4. Record baseline benchmarks and wire regression gates.
5. Add RaptorQ sidecar durability for persistent evidence artifacts.

## Porting Artifact Set

- PLAN_TO_PORT_PYTORCH_TO_RUST.md
- EXISTING_PYTORCH_STRUCTURE.md
- PROPOSED_ARCHITECTURE.md
- FEATURE_PARITY.md

These four docs are now the canonical porting-to-rust workflow for this repo.
