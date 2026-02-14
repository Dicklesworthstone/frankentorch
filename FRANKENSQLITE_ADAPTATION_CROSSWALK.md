# FrankenSQLite-to-FrankenTorch Adaptation Crosswalk

Normative exemplar copied into this repository:
- `COMPREHENSIVE_SPEC_FOR_FRANKENSQLITE_V1_REFERENCE.md` (846,244 bytes)

This document maps proven FrankenSQLite patterns onto FrankenTorch's deterministic autograd mission.

Absolute doctrine carried from this project into all adaptations:
- FrankenTorch targets total feature/functionality overlap for true drop-in replacement behavior.
- Packetized sequencing controls execution risk but does not permit permanent exclusions.

## 1. Direct Pattern Reuse

| FrankenSQLite Pattern | FrankenTorch Adaptation |
|---|---|
| RaptorQ-everywhere artifact durability | RaptorQ sidecars for conformance fixtures, gradient mismatch bundles, and benchmark baselines |
| strict vs hardened mode split | strict: PyTorch-observable parity; hardened: bounded defenses with fail-closed incompatibility handling |
| evidence ledger for consequential runtime decisions | dispatch decision ledger + backward replay ledger for DAC proofs |
| conformance harness as release gate | differential oracle harness against `legacy_pytorch_code/pytorch` with machine-readable parity reports |
| one-lever optimization doctrine + isomorphism proofs | each performance change must include gradient-isomorphism proof and benchmark delta |

## 2. Asupersync/FrankenTUI Leverage Plan

### Asupersync
- Integrate `Cx`/budgeted execution for long-running conformance and benchmark jobs.
- Reuse asupersync evidence and oracle patterns for deterministic replay reports.
- Bind RaptorQ sidecar pipelines to asupersync transport/codec surfaces (mirroring FrankenSQLite usage).

### FrankenTUI (`ftui`)
- Build an operator-facing parity cockpit:
  - live strict/hardened drift summaries
  - gradient mismatch drilldown
  - decode-proof and scrub status panels
- Keep TUI as observability/control plane, not core execution dependency.

## 3. Mandatory Doctrine Mapping

| Doctrine | FrankenTorch Execution Rule |
|---|---|
| alien-artifact-coding | every adaptive decision uses explicit loss model and evidence terms |
| alien-graveyard | candidate algorithm must pass score gate (Impact*Confidence/Effort >= 2.0) |
| extreme-software-optimization | baseline -> profile -> one lever -> isomorphism proof -> re-baseline |
| RaptorQ durability | durable artifact classes always emit sidecar manifest + scrub + decode proof chain |

## 4. Immediate Next Execution Packets

1. `FT-P2C-002`: dispatch key ordering and fallback semantics.
2. `FT-P2C-003`: schema ingestion + op registration parity.
3. `FT-P2C-004`: autograd scheduler state machine extraction.
4. `FT-P2C-006`: checkpoint format parity and fail-closed unknown-field policy.
