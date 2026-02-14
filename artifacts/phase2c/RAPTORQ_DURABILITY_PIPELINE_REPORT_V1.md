# RaptorQ Durability Pipeline Report (v1)

Date: 2026-02-14  
Bead: `bd-3v0.9`

## Scope

Automate durable sidecar/proof generation and validation for:
- conformance fixture bundle
- benchmark baseline bundle
- migration manifest
- reproducibility ledger
- long-lived state snapshot
- packet parity reports (`FT-P2C-001`, `FT-P2C-002`, `FT-P2C-004`, `FT-P2C-006`)

Command:
- `cargo run -p ft-conformance --bin run_raptorq_durability_pipeline`

## Produced Artifacts

- `artifacts/phase2c/RAPTORQ_REPAIR_SYMBOL_MANIFEST_V1.json`
- `artifacts/phase2c/RAPTORQ_INTEGRITY_SCRUB_REPORT_V1.json`
- `artifacts/phase2c/RAPTORQ_DECODE_PROOF_EVENTS_V1.json`
- `artifacts/phase2c/raptorq_sidecars/` (global durable artifact sidecars/proofs)
- packet-local updates:
  - `artifacts/phase2c/FT-P2C-001/parity_report.raptorq.json`
  - `artifacts/phase2c/FT-P2C-001/parity_report.decode_proof.json`
  - `artifacts/phase2c/FT-P2C-002/parity_report.raptorq.json`
  - `artifacts/phase2c/FT-P2C-002/parity_report.decode_proof.json`
  - `artifacts/phase2c/FT-P2C-004/parity_report.raptorq.json`
  - `artifacts/phase2c/FT-P2C-004/parity_report.decode_proof.json`
  - `artifacts/phase2c/FT-P2C-006/parity_report.raptorq.json`
  - `artifacts/phase2c/FT-P2C-006/parity_report.decode_proof.json`

## Outcome Summary

- targets processed: `9`
- sidecars emitted: `9`
- scrub failures: `0`
- decode proof events: `9`
- corruption probe passes: `9`

## Gate Integration

- `validate_phase2c_artifacts` now enforces presence/schema/health of:
  - `RAPTORQ_REPAIR_SYMBOL_MANIFEST_V1.json`
  - `RAPTORQ_INTEGRITY_SCRUB_REPORT_V1.json`
  - `RAPTORQ_DECODE_PROOF_EVENTS_V1.json`
- Gate assertions include:
  - manifest `failed_targets == 0`
  - scrub `failed == 0`
  - decode events `corruption_probe_passed == total_events`

## Method-Stack Alignment

- alien-artifact-coding: deterministic durability evidence and replay-ready decode-event ledger.
- extreme-software-optimization: one lever (pipeline automation) without behavior changes in runtime kernels/autograd.
- RaptorQ-everywhere durability: repair-symbol manifest + integrity scrub + decode-proof event artifacts emitted and validated.
- frankenlibc/frankenfs security-compatibility doctrine: fail-closed global gate now blocks READY state on durability-report drift.
