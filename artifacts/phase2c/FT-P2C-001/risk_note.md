# FT-P2C-001 — Risk Note

## Primary Risk

Autograd drift under future optimization pressure (dispatch changes or in-place mutation semantics).

## Risk Tier

Critical.

## Mitigations Implemented

1. Deterministic backward traversal and replayable step trace (`BackwardReport.steps`).
2. Explicit compatibility checks before kernel execution (`dtype/device`).
3. Mode-aware dispatch decision records in evidence ledger.
4. Strict + hardened conformance execution for the same fixture family.

## Residual Risk

- Current fixture family is scalar-only; vector/tensor shape interactions are not yet covered.
- Storage alias/view semantics not yet implemented in this packet.

## Next Controls

- Add tensor-shape and alias/version fixtures in `FT-P2C-002`.
- Add adversarial mutation fixtures in `FT-P2C-004`.
- Add RaptorQ sidecar generation for parity reports once packet report serialization lands.

