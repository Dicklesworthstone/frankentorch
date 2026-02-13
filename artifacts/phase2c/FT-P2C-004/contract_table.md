# FT-P2C-004 — Contract Table

## Autograd Scheduling Contract (Scoped)

| Contract ID | Input | Output | Error | Invariant |
|---|---|---|---|---|
| `AUTOGRAD-SCHED-001` | rooted graph + options | dependency map | `UnknownNode` | dependencies count outstanding downstream uses |
| `AUTOGRAD-SCHED-002` | ready tasks | deterministic execution order | `DependencyUnderflow` | max-heap tie-break (`higher NodeId first`) is stable |
| `AUTOGRAD-SCHED-003` | rooted graph + gradients | `BackwardReport` | `UnknownNode` | gradients are deterministic across replays |
| `AUTOGRAD-SCHED-004` | reentrant depth + strict policy | failure | `ReentrantDepthExceeded` | strict mode never applies recovery |
| `AUTOGRAD-SCHED-005` | reentrant depth + hardened policy | bounded fallback + telemetry | none | hardened fallback sets `reentrant_guard_triggered=true` |

## Strict vs Hardened Policy

| Mode | Behavior |
|---|---|
| strict | reentrant overflow fails closed before scheduler execution |
| hardened | reentrant overflow is clamped to configured bound and recorded in telemetry |

## Scheduler Telemetry Contract

`BackwardReport.telemetry` carries:
- `execution_order`
- `queue_pushes`
- `queue_pops`
- `max_queue_len`
- `dependency_snapshot`
- `reentrant_depth`
- `reentrant_guard_triggered`
- `hardened_fallback_used`

These fields are deterministic for identical inputs and options.
