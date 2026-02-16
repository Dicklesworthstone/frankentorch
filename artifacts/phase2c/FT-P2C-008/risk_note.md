# FT-P2C-008 — Risk Note

## Primary Risk

NN module/state compatibility drift that could silently accept malformed load payloads, mutate state keysets, or bypass strict fail-closed behavior under prefix/hook pathways.

## Risk Tier

High.

## Mitigations Implemented

1. Strict/hardened packet parity tests for registration, state export/load, mode propagation, prefix normalization, hooks, and assign-path guards.
2. Unit/property evidence for packet-008 state invariants and structured forensic logging:
   - `artifacts/phase2c/FT-P2C-008/unit_property_quality_report_v1.json`
3. Differential/metamorphic/adversarial packet checks with explicit hardened allowlist posture:
   - `artifacts/phase2c/FT-P2C-008/differential_packet_report_v1.json`
   - `artifacts/phase2c/FT-P2C-008/differential_reconciliation_v1.md`
4. Packet-filtered E2E replay/forensics evidence with deterministic scenario IDs and seed capture:
   - `artifacts/phase2c/e2e_forensics/ft-p2c-008.jsonl`
   - `artifacts/phase2c/FT-P2C-008/e2e_replay_forensics_linkage_v1.json`
5. Optimization/isomorphism gate confirms no behavior drift while reducing packet E2E latency tails:
   - `artifacts/phase2c/FT-P2C-008/optimization_delta_v1.json`
   - `artifacts/phase2c/FT-P2C-008/optimization_isomorphism_v1.md`
6. Packet parity sidecar/decode-proof durability anchors:
   - `artifacts/phase2c/FT-P2C-008/parity_report.raptorq.json`
   - `artifacts/phase2c/FT-P2C-008/parity_report.decode_proof.json`

## Residual Risk

- Hardened non-strict missing/unexpected key handling remains allowlisted (`nn_state.non_strict_missing_unexpected`) and must stay bounded to non-incompatible shape cases.
- Broader module ecosystem parity (distributed/quantization/export wrappers) remains explicitly out of scope for this packet.

## Next Controls

1. Keep packet-level allowlist drift constrained and audit any additions against threat model `T008-03`/`T008-04`.
2. Refresh packet RaptorQ scrub/decode manifests whenever parity evidence changes.
3. Close packet final evidence gate with validator + readiness drill sign-off.
