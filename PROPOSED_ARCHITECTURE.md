# PROPOSED_ARCHITECTURE

## 1. Architecture Principles

1. Spec-first implementation; no line translation.
2. Strict mode for compatibility; hardened mode for defensive operation.
3. RaptorQ sidecars for long-lived conformance and benchmark artifacts.
4. Profile-first optimization with isomorphism proofs.
5. Deterministic Autograd Contract (DAC): replayable backward traces with evidence ledger entries.

## 2. Crate Map

- `ft-core`: tensor metadata/value/version model and compatibility invariants.
- `ft-dispatch`: scoped op routing and dispatch decision records.
- `ft-kernel-cpu`: eager CPU kernels (first wave).
- `ft-autograd`: graph capture + deterministic backward replay.
- `ft-device`: device guards and same-device contracts.
- `ft-serialize`: snapshot/checkpoint codecs.
- `ft-runtime`: mode policy + evidence ledger + durability envelopes.
- `ft-api`: user-facing execution session.
- `ft-conformance`: differential fixtures, parity checks, and benchmark harness.

## 3. Runtime Flow

1. API layer validates request context and mode.
2. Dispatcher selects kernel and emits decision evidence.
3. Kernel executes with explicit compatibility checks.
4. Autograd engine builds replayable trace and deterministic backward report.
5. Conformance layer compares against oracle fixtures and emits parity artifacts.
6. Durability layer wraps long-lived artifacts with sidecar metadata.

## 4. Asupersync + FrankenTUI Leverage

- `asupersync`: planned execution budget/cancellation/evidence plumbing for long-running conformance and benchmark jobs.
- `ftui`: planned parity cockpit for strict/hardened drift inspection and artifact health.
- Both are wired via `ft-runtime` feature gates:
  - `asupersync-integration`
  - `frankentui-integration`

## 5. Compatibility and Security

- strict mode: maximize scoped PyTorch parity.
- hardened mode: same outward contract with defensive checks and audited diagnostics.
- unknown incompatible metadata/protocol fields: fail-closed.

## 6. Performance Contract

- baseline -> profile -> one-lever optimization -> conformance proof -> re-baseline.
- track p50/p95/p99 plus memory churn deltas.

## 7. Conformance Contract

- fixture families are versioned and machine-readable.
- strict and hardened mode both required for green status.
- parity reports are release-gating for scoped API families.
