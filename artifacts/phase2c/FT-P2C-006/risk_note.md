# FT-P2C-006 — Risk Note

## Primary Risk

Checkpoint compatibility drift or silent payload corruption.

## Risk Tier

High.

## Mitigations Implemented

1. Version gate with explicit mismatch rejection.
2. Unknown-field fail-closed behavior in strict mode.
3. Hardened mode diagnostics are bounded and still fail-closed on incompatibility.
4. Deterministic checksum field over normalized payload.
5. RaptorQ sidecar generation + decode proof with exact-byte recovery check.

## Residual Risk

- Scoped envelope is JSON-based and intentionally narrower than PyTorch's full binary archive semantics.
- Multi-storage and alias graph checkpoint fidelity is out of current scope.

## Next Controls

1. Add binary-compat fixtures for larger tensor payload classes.
2. Add adversarial mutation corpus for checksum and unknown-field fuzzing.
3. Add periodic durability scrub job artifact for long-lived snapshots.
