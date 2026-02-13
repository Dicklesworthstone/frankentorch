# Scalar DAC Optimization Artifact (2026-02-13)

## Baseline Command

- `cargo test -p ft-conformance microbench_produces_percentiles -- --nocapture`

## Baseline Snapshot (ns)

- p50: `2655`
- p95: `3517`
- p99: `3517`
- mean: `4651`

## Opportunity Matrix

| Hotspot | Impact (1-5) | Confidence (1-5) | Effort (1-5) | Score | Decision |
|---|---:|---:|---:|---:|---|
| backward traversal order and allocation behavior | 3 | 4 | 2 | 6.0 | Implement now |
| symbolic-shape plumbing | 2 | 2 | 5 | 0.8 | Defer |
| dynamic kernel table abstraction | 2 | 3 | 4 | 1.5 | Defer |

## One Lever Implemented

- Replace recursive backward with deterministic reverse-index traversal over a preallocated gradient vector.

## Isomorphism Proof

- Ordering preserved: yes (reverse topological order induced by construction order).
- Tie-breaking unchanged: yes (single deterministic order by node id).
- Floating-point: identical arithmetic expressions for add/mul chain rule.
- RNG seeds: N/A (no RNG in path).
- Golden outputs:
  - `sha256sum -c artifacts/optimization/golden_checksums.txt` passes.
  - fixture suite validated in strict and hardened modes.

## Fallback

- If traversal change regresses parity, revert to previous traversal strategy and gate on fixture diff.
