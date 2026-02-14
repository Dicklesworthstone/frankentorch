# Failure Forensics Envelope Schema V1

Bead: `bd-3v0.21`  
Schema ID: `ft-failure-forensics-index-v1`

## Purpose

Provide a stable, human-readable and machine-consumable envelope for failed scenarios so maintainers can go from failure signal to replay command with no hidden steps.

## Top-Level Index Object

Required keys:
- `schema_version`
- `generated_unix_ms`
- `source_artifacts`
- `summary`
- `suite_evidence_templates`
- `failures`
- `triaged_incidents`

## Source Artifacts

`source_artifacts` must include:
- `e2e_log_path`
- `triage_path`
- `differential_report_path`

## Failure Envelope (`failures[]`)

Required keys:
- `failure_id`: deterministic hash ID (`ffx64:*`)
- `packet_id`
- `suite_id`
- `scenario_id`
- `mode`
- `reason_code`
- `fixture_id`
- `replay_command`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `evidence_links`

## Evidence Link (`evidence_links[]`)

Required keys:
- `category` (`unit_property`, `differential`, `e2e`, `performance`, `raptorq`)
- `path_or_ref`
- `exists`

## Behavioral Requirements

1. `scenario_id` must remain compatible with `scenario_id` contract in `crates/ft-conformance/src/lib.rs:2345`.
2. `replay_command` and `env_fingerprint` must be present for every failed scenario.
3. `evidence_links` must include at least one entry for each evidence category:
   - unit/property
   - differential
   - e2e
   - performance
   - RaptorQ
4. Envelope generation must be deterministic for equal inputs (stable `failure_id` hashing).
5. Forensics UX must allow direct triage from one JSON object without requiring manual file discovery.

## Current Generator

Binary:
- `crates/ft-conformance/src/bin/build_failure_forensics_index.rs`

Example command:
- `cargo run -p ft-conformance --bin build_failure_forensics_index -- --e2e artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl --triage artifacts/phase2c/e2e_forensics/crash_triage_full_v1.json --output artifacts/phase2c/e2e_forensics/failure_forensics_index_v1.json`
