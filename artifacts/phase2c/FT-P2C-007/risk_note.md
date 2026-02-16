# FT-P2C-007 — Risk Note

## Primary Risk

Dispatch/device-compatibility drift that could silently permit unsafe backend transitions or cross-device execution.

## Risk Tier

High.

## Mitigations Implemented

1. Strict/hardened packet parity tests for dispatch routes and fail-closed keyset/device/dtype mismatches.
2. Unit/property evidence for device-guard and cross-device rejection behavior:
   - `artifacts/phase2c/FT-P2C-007/unit_property_quality_report_v1.json`
3. Differential packet checks with explicit hardened allowlist posture for bounded composite fallback:
   - `artifacts/phase2c/FT-P2C-007/differential_packet_report_v1.json`
   - `artifacts/phase2c/FT-P2C-007/differential_reconciliation_v1.md`
4. Packet-filtered E2E replay/forensics evidence with deterministic seeds and scenario IDs:
   - `artifacts/phase2c/e2e_forensics/ft-p2c-007.jsonl`
   - `artifacts/phase2c/FT-P2C-007/e2e_replay_forensics_linkage_v1.json`
5. Optimization/isomorphism gate confirms no behavior drift while reducing packet E2E latency tails:
   - `artifacts/phase2c/FT-P2C-007/optimization_delta_v1.json`
   - `artifacts/phase2c/FT-P2C-007/optimization_isomorphism_v1.md`
6. Packet parity sidecar/decode proof durability anchors:
   - `artifacts/phase2c/FT-P2C-007/parity_report.raptorq.json`
   - `artifacts/phase2c/FT-P2C-007/parity_report.decode_proof.json`

## Residual Risk

- Non-CPU backend families remain explicitly out of scope for this packet and are tracked via deferred gaps.
- Hardened composite fallback remains allowlisted and requires continued monitoring for scope creep.

## Next Controls

1. Expand packet-008 contract/execution coverage to close non-CPU backend domain gaps.
2. Add deeper adversarial backend-transition fixtures (mixed backend families and malformed keyset variants).
3. Keep packet-level RaptorQ scrub/decode reports refreshed whenever parity artifacts change.
