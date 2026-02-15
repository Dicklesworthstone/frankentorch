# FT-P2C-006 Differential Drift Reconciliation (v1)

## Scope

- source report: `artifacts/phase2c/conformance/differential_report_v1.json`
- packet slice report: `artifacts/phase2c/FT-P2C-006/differential_packet_report_v1.json`
- packet: `FT-P2C-006`

## Result

- packet checks: `36`
- pass checks: `34`
- non-pass checks: `2`
- packet allowlisted drifts: `2`
- packet blocking drifts: `0`

Packet allowlisted drifts are both scoped to hardened bounded malformed diagnostics:
- `serialization.bounded_malformed_diagnostic` (`hardened:checkpoint_basic`, comparator `policy`)
- `serialization.bounded_malformed_diagnostic` (`hardened:checkpoint_sparse_grad`, comparator `policy`)

## Metamorphic + Adversarial Coverage

Metamorphic comparator executed under strict and hardened modes:
- `metamorphic_entry_order_hash_invariant`

Adversarial fail-closed comparators executed under strict and hardened modes:
- `adversarial_unknown_field_rejected`
- `adversarial_version_mismatch_rejected`
- `adversarial_checksum_tamper_rejected`
- `adversarial_malformed_json_rejected`
- `adversarial_raptorq_corruption_probe`

## Risk-Note Linkage

Validated threat controls mapped in `artifacts/phase2c/FT-P2C-006/risk_note.md` and `artifacts/phase2c/FT-P2C-006/threat_model.md`:
- `T006-01`: unknown-field compatibility confusion fails closed in strict+hardened modes
- `T006-02`: schema/version drift fails closed
- `T006-03`: checksum tamper/replay mismatch fails closed
- `T006-04`: malformed payload rejection in both modes; bounded diagnostic metadata only in hardened mode
- `T006-05`: corruption probe changes RaptorQ proof/hash surface (or fails closed), preserving detection posture

Deferred scope remains explicit:
- `serialization:multi_storage_alias_graph_gap`

## Method-Stack Status for This Bead

- alien-artifact-coding: deterministic FT-P2C-006 differential slice emitted with explicit mode-split policy classification and scenario linkage.
- extreme-software-optimization: no optimization lever changed; checksum/proof behavior-isomorphism preserved for non-allowlisted comparators.
- security/compatibility doctrine: strict mode remains fully fail-closed; hardened divergence is confined to allowlisted bounded malformed diagnostics.
- RaptorQ-everywhere durability: packet differential adversarial checks include corruption-probe comparator; sidecar durability pipeline remains handled by durability/evidence beads.
