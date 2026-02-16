# Security + Compatibility Threat Matrix v1

Version: `security-compat-matrix-v1`  
Scope: Phase-2C packet families (`FT-P2C-001`, `FT-P2C-002`, `FT-P2C-003`, `FT-P2C-004`, `FT-P2C-005`, `FT-P2C-006`)

## Policy Core

- strict mode:
  - maximize behavioral compatibility for scoped contracts.
  - fail-closed on unknown/incompatible inputs.
  - no behavior-altering recovery.
- hardened mode:
  - preserve external API contract while allowing bounded, explicit defensive handling.
  - deviations must be allowlisted ahead of time; ad-hoc recovery is forbidden.

## Threat Classes

| Threat Class | Example Attack/Failure | Primary Impact | Baseline Mitigation |
|---|---|---|---|
| Compatibility confusion | unknown key/field/version accepted silently | semantic drift, unsafe behavior | explicit fail-closed gate + schema/version checks |
| Gradient corruption | incorrect backward scheduling/order | model correctness failure | deterministic scheduler + invariant/conformance checks |
| Dispatch misrouting | wrong kernel/backend path selection | wrong outputs, hidden drift | explicit key precedence and route evidence ledger |
| Serialization mismatch | malformed or tampered checkpoint accepted | replay inconsistencies | deterministic hash + strict parser + sidecar verification |
| Recovery-path ambiguity | unproven repair/decode path | unverifiable data integrity | decode proof artifacts + scrub requirements |

## Packet Matrix

| Packet | High-Risk Surface | Strict Mode Mitigation | Hardened Mode Allowlisted Deviations |
|---|---|---|---|
| `FT-P2C-001` | tensor metadata/versioning | reject incompatible metadata states | none (parity-only) |
| `FT-P2C-002` | dispatch key routing | reject unknown bits and incompatible keysets | composite/backend-select fallback to backend key only |
| `FT-P2C-003` | op schema ingestion | reject malformed/ambiguous schema strings and incompatible dispatch metadata | none (parity-only) |
| `FT-P2C-004` | autograd scheduling/reentrancy | reentrant depth overflow fails | bounded depth clamp with explicit telemetry flag |
| `FT-P2C-005` | CPU kernel first-wave semantics and packet projection drift | reject dtype/device mismatch, invalid keysets, and incompatible tensor-meta states | projected dispatch fallback only (`dispatch.composite_backend_fallback`); contiguous projection drift remains blocking until resolved |
| `FT-P2C-006` | checkpoint parsing/recovery | unknown field/version/hash mismatch fails closed | bounded malformed diagnostic context only (`serialization.bounded_malformed_diagnostic`); no incompatible acceptance |

## Drift Gates

1. Any non-allowlisted hardened deviation is a release-blocking compatibility failure.
2. Unknown incompatible feature path must fail closed in both modes.
3. Security-sensitive recovery must emit deterministic evidence artifacts.
4. Packet `NOT_READY` if mandatory schema or allowlist controls are missing.

## Required Artifacts

- `artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json`
- packet `risk_note.md` files with compatibility/security mitigations
- packet threat-model artifacts for active packet C tasks (current: `artifacts/phase2c/FT-P2C-006/threat_model.md`)
- packet parity sidecars and decode proofs
