# FT-P2C-003 Differential Drift Reconciliation (v1)

## Scope

- source report: `artifacts/phase2c/conformance/differential_report_v1.json`
- packet slice report: `artifacts/phase2c/FT-P2C-003/differential_packet_report_v1.json`
- packet: `FT-P2C-003`

## Result

- packet checks: `38`
- pass checks: `38`
- non-pass checks: `0`
- packet allowlisted drifts: `0`
- packet blocking drifts: `0`

Global report includes `2` allowlisted drifts, both outside this packet:
- `dispatch.composite_backend_fallback` (`FT-P2C-002`)
- `autograd.reentrant_depth_bounded_fallback` (`FT-P2C-004`)

## Metamorphic + Adversarial Coverage

Metamorphic comparator executed under strict and hardened modes:
- `metamorphic_name_normalization`

Adversarial fail-closed scenarios executed under strict and hardened modes:
- `malformed_schema_rejected`
- `dispatch_metadata_incompatible`

## Risk-Note Linkage

Validated threat controls mapped in `artifacts/phase2c/FT-P2C-003/risk_note.md`:
- `THR-301`: malformed schema token stream fail-closed behavior
- `THR-302`: operator-name normalization determinism
- `THR-303`: out/structured alias contract preservation
- `THR-304`: dispatch metadata incompatibility fail-closed behavior
- `THR-305`: replay evidence contract continuity through differential checks

Modeled deferred gap:
- `GAP-SCHEMA-001` symbolic-shape semantics remain deferred, but `frankentorch-795j` adds the active `symbolic_shape_gap_marker` fixture so strict+hardened e2e/differential logs fail closed with an explicit gap marker instead of silently accepting symbolic-shape behavior.

## Method-Stack Status for This Bead

- alien-artifact-coding: deterministic differential packet slice emitted with explicit threat linkage and scenario IDs.
- extreme-software-optimization: no optimization lever changed; behavior-isomorphism preserved with zero packet drift.
- frankenlibc/frankenfs security-compatibility doctrine: strict+hardened mode checks remain fail-closed for malformed schema and incompatible dispatch metadata.
- RaptorQ-everywhere durability: sidecar emission remains deferred to packet final evidence bead (`bd-3v0.14.9`).
