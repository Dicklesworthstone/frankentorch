# METHOD_STACK_REPORT_2026-02-14

Scope: structured logging contract uplift + e2e forensics emitter + validator parallel optimization.

## alien-artifact-coding

- Added explicit structured evidence schema (`ft-conformance-log-v1`) with deterministic `scenario_id`, `seed`, and replay fields.
- Added machine-readable e2e forensic JSONL output to make decision traces auditable and replay-ready.
- Added contract document: `artifacts/phase2c/TEST_LOG_CONTRACT_V1.md`.

## extreme-software-optimization

- Profile target: `validate_phase2c_artifacts` on synthetic large corpus.
- Single retained lever: packet-level parallel validation.
- Measured improvement: `63.9 ms` -> `35.1 ms` mean (`1.82x` faster) with deterministic ordering preserved.
- Evidence artifact: `artifacts/optimization/2026-02-14_packet_parallel_validation.md`.
- Rebaseline artifact (step tails + memory + backward-included microbench):
  - `artifacts/optimization/2026-02-14_foundation_perf_rebaseline.json`
  - `artifacts/optimization/2026-02-14_foundation_perf_rebaseline.md`

## alien-graveyard

- Applied high-EV primitive: independent task decomposition + parallel execution at packet boundary.
- Adoption wedge: env-controlled fallback (`FT_DISABLE_PACKET_PARALLELISM=1`) for conservative mode.
- Behavior-proof posture: output sort preserved; validator tests + workspace gates green post-change.

## RaptorQ-everywhere durability

- Added automated durability pipeline: `cargo run -p ft-conformance --bin run_raptorq_durability_pipeline`.
- Emitted durable artifact set:
  - `artifacts/phase2c/RAPTORQ_REPAIR_SYMBOL_MANIFEST_V1.json`
  - `artifacts/phase2c/RAPTORQ_INTEGRITY_SCRUB_REPORT_V1.json`
  - `artifacts/phase2c/RAPTORQ_DECODE_PROOF_EVENTS_V1.json`
- Added validator gate checks in `validate_phase2c_artifacts` for global durability schema + health.
- Corruption probe posture: decode-event summary enforces `corruption_probe_passed == total_events`.
