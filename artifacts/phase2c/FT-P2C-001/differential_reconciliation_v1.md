# FT-P2C-001 Differential Drift Reconciliation (v1)

## Scope

- source report: `artifacts/phase2c/conformance/differential_report_v1.json`
- packet slice report: `artifacts/phase2c/FT-P2C-001/differential_packet_report_v1.json`
- packet: `FT-P2C-001`

## Result

- packet checks: `86`
- pass checks: `86`
- non-pass checks: `0`
- packet allowlisted drifts: `0`
- packet blocking drifts: `0`

Global report includes `2` allowlisted drifts, both outside this packet:
- `dispatch.composite_backend_fallback` (`FT-P2C-002`)
- `autograd.reentrant_depth_bounded_fallback` (`FT-P2C-004`)

## Metamorphic + Adversarial Coverage

Metamorphic comparators executed under strict and hardened modes:
- `metamorphic_offset_shift_numel_local`
- `metamorphic_offset_shift_linear_local`
- `metamorphic_offset_shift_contiguous_local`
- `metamorphic_offset_shift_numel_oracle`
- `metamorphic_offset_shift_linear_oracle`
- `metamorphic_offset_shift_contiguous_oracle`

Adversarial fail-closed scenarios executed under strict and hardened modes:
- `invalid_rank_stride_mismatch`
- `invalid_storage_offset_overflow`

## Risk-Note Linkage

Validated threat controls mapped in `artifacts/phase2c/FT-P2C-001/risk_note.md`:
- `THR-001`: rank/stride mismatch fail-closed
- `THR-002`: storage-offset overflow fail-closed
- `THR-004`: forensic/replay evidence chain integrity

Remaining deferred gap:
- `GAP-UX-001` (`THR-005` dtype/device mismatch adversarial fixture candidate)

## Method-Stack Status for This Bead

- alien-artifact-coding: deterministic report + explicit risk linkage ledger emitted.
- extreme-software-optimization: no optimization lever changed; behavior-isomorphism preserved via zero-drift differential pass.
- RaptorQ-everywhere durability: deferred to packet final evidence bead (`bd-3v0.12.9`) where sidecar generation is consolidated.
- frankenlibc/frankenfs security-compatibility doctrine: strict/hardened split validated with fail-closed adversarial coverage and no unallowlisted drift.
