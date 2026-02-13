# FT-P2C-002 — Risk Note

## Primary Risk

Dispatch precedence drift causing wrong kernel route selection under strict mode.

## Risk Tier

Critical.

## Mitigations Implemented

1. Explicit keyset validation with unknown-bit fail-closed behavior.
2. Separate type-priority and backend-priority resolution helpers.
3. Mode-split policy:
   - strict rejects composite/backend-select fallback,
   - hardened allows bounded fallback with explicit evidence flag.
4. Conformance fixture includes explicit strict-vs-hardened divergence case.

## Residual Risk

- Key domain is intentionally scoped to CPU + autograd CPU vertical slice.
- Upstream enum ordering in PyTorch can evolve; parity drift risk remains for unscoped keys.

## Next Controls

1. Add cross-backend key families in `FT-P2C-007`.
2. Add schema-ingested operator routing matrices in `FT-P2C-003`.
3. Extend fixture family with adversarial raw-bitset corruption probes.
