# FT-P2C-008 Differential Drift Reconciliation (v1)

## Scope

- source report: `artifacts/phase2c/conformance/differential_report_v1.json`
- packet slice report: `artifacts/phase2c/FT-P2C-008/differential_packet_report_v1.json`
- packet: `FT-P2C-008`

## Result

- packet checks: `68`
- pass checks: `67`
- non-pass checks: `1`
- packet allowlisted drifts: `1`
- packet blocking drifts: `0`

Packet drift is explicitly allowlisted and bounded to hardened mode policy split for non-strict load semantics:
- drift id: `nn_state.non_strict_missing_unexpected`
- scenario: `nn_state/hardened:load_state_missing_unexpected_mode_split`
- comparator: `policy`
- strict counterpart remains `pass` with `strict_fail_closed_mode_split` evidence

## Metamorphic + Adversarial Coverage

Metamorphic comparators executed in strict and hardened modes:
- `metamorphic_state_export_order_invariant`
- `metamorphic_mode_transition_idempotent`
- `metamorphic_prefix_normalization_idempotent`
- `metamorphic_hook_trace_stable`

Adversarial comparators executed in strict and hardened modes:
- `adversarial_invalid_registration_name_rejected`
- `adversarial_incompatible_shape_rejected`
- `adversarial_assign_shape_rejected`
- `adversarial_non_persistent_buffer_excluded`

## Risk-Note Linkage

Validated controls from `artifacts/phase2c/FT-P2C-008/threat_model.md`:
- `T008-03`: strictness bypass remains fail-closed in strict mode; hardened non-strict missing/unexpected behavior is explicitly allowlisted and auditable.
- `T008-04`: prefix normalization metamorphic idempotence now holds in both modes.
- `T008-05`: hook trace stability remains deterministic under metamorphic validation.
- `T008-07`: incompatible-shape and assign-path adversarial checks remain fail-closed.

Allowlist governance is anchored in:
- `artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json`

## Method-Stack Status for This Bead

- alien-artifact-coding: deterministic packet-scoped differential evidence emitted with scenario-level mode split classification.
- extreme-software-optimization: no optimization lever changed; this bead validates behavior only.
- security/compatibility doctrine: strict mode is fail-closed; hardened divergence is bounded to one allowlisted policy drift.
- RaptorQ-everywhere durability: differential evidence is complete for packet F scope; sidecar/decode closure remains in downstream evidence beads.
