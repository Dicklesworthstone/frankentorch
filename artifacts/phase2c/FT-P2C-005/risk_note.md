# FT-P2C-005 — Risk Note

## Primary Risk

Compatibility drift in CPU-kernel first-wave semantics across projected scalar, tensor-meta, and dispatch pathways.

## Risk Tier

High.

## Mitigations Implemented

1. Deterministic scalar arithmetic validation for add/mul core paths.
2. Fail-closed dtype/device compatibility enforcement in kernel dispatch.
3. Dispatch-key parsing and mode-split policy controls with explicit hardened allowlist linkage to `FT-P2C-002`.
4. Structured replay/forensics envelope with deterministic scenario IDs, seeds, and artifact refs:
   - `artifacts/phase2c/e2e_forensics/ft-p2c-005.jsonl`
5. Differential/metamorphic/adversarial packet projection:
   - `artifacts/phase2c/FT-P2C-005/differential_packet_report_v1.json`
   - `artifacts/phase2c/FT-P2C-005/differential_reconciliation_v1.md`
6. Threat-model controls and abuse classes:
   - `artifacts/phase2c/FT-P2C-005/threat_model.md`

## Current Drift Posture

- Allowlisted drifts: `8` (hardened dispatch mode-split drift `dispatch.composite_backend_fallback`, projected from `FT-P2C-002`).
- Blocking drifts: `2` (`tensor_meta.contiguous_mismatch` for `broadcast_stride_zero_valid`, strict + hardened).

## Residual Risk

- Packet remains blocked for full parity closure while `tensor_meta.contiguous_mismatch` remains unresolved in FT-P2C-005 projection.
- First-wave scope does not yet cover full TensorIterator dtype/vectorized/sparse parity surface.

## Next Controls

1. Resolve the blocking tensor-meta contiguous drift (`broadcast_stride_zero_valid`) in projected packet checks.
2. Re-run packet differential projection and require zero blocking drifts before parity-green promotion.
3. Carry resolved state into FT-P2C-007 dependency chain and readiness-gate evidence.
