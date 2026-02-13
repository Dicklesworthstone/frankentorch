# Phase-2C Hotspot Matrix (2026-02-13)

Baseline command set:
- `CARGO_TARGET_DIR=target_codex cargo test -p ft-conformance -- --nocapture`
- microbench line: `microbench_ns p50=3737 p95=4869 p99=4869 mean=6248`

## Opportunity Matrix

| Hotspot | Impact (1-5) | Confidence (1-5) | Effort (1-5) | Score | Decision |
|---|---:|---:|---:|---:|---|
| Dispatch key resolution (`DispatchKeySet::highest_priority_type_id`) | 3 | 4 | 2 | 6.0 | implemented this pass |
| Autograd backward scheduling (`Tape::backward_with_options`) | 5 | 4 | 3 | 6.7 | implemented this pass |
| Serialization durability path (`generate_raptorq_sidecar`) | 4 | 4 | 3 | 5.3 | implemented this pass |
| Multi-threaded autograd queueing | 4 | 2 | 5 | 1.6 | deferred (below score threshold) |
| Full `.pt` archive binary compatibility | 5 | 2 | 5 | 2.0 | deferred to dedicated packet scope |

## One-Lever Discipline Record

1. Packet `FT-P2C-002`: dispatch key model and routing contract only.
2. Packet `FT-P2C-004`: scheduler/dependency/reentrancy contract only.
3. Packet `FT-P2C-006`: checkpoint+durability contract only.
