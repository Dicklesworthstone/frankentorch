# FT-P2C-004 — Risk Note

## Primary Risk

Gradient drift from scheduler ordering or dependency accounting errors.

## Risk Tier

Critical.

## Mitigations Implemented

1. Explicit dependency counting before queue execution.
2. Deterministic ready-queue ordering and execution-order telemetry.
3. Strict reentrant overflow fails closed.
4. Hardened reentrant overflow path is bounded and explicitly marked.
5. Conformance fixture validates gradients + execution order + mode-split reentrancy behavior.

## Residual Risk

- This packet is single-threaded and CPU-only; concurrency-specific race classes are out of scope.
- Full PyTorch graph task/future synchronization behavior remains unimplemented.

## Next Controls

1. Add multi-branch and shared-subgraph adversarial fixtures.
2. Add scheduler p95/p99 telemetry baselines over larger traces.
3. Add replay hash witness for execution-order vectors.
