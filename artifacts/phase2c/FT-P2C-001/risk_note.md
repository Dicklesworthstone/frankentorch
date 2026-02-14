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
5. Tensor-meta differential layer now includes metamorphic offset-shift checks and adversarial oracle fail-closed checks with resource guards.

## Residual Risk

- Tensor metadata fixtures still under-sample high-rank and symbolic shape families.
- Oracle guard intentionally skips unsafe huge-allocation adversarial probes; those remain covered by local fail-closed checks only.

## Next Controls

- Expand tensor-meta adversarial corpus with seeded fuzz-style case generation and bounded oracle mirror probes.
- Add symbolic-shape parity coverage in `FT-P2C-002`.
- Add mutation-path adversarial fixtures in `FT-P2C-004`.
