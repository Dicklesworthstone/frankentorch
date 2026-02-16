# FT-P2C-007 Differential Drift Reconciliation (v1)

## Scope

- source report: `artifacts/phase2c/conformance/differential_report_v1.json`
- packet slice report: `artifacts/phase2c/FT-P2C-007/differential_packet_report_v1.json`
- packet: `FT-P2C-007`
- projection basis: `dispatch_key` differential suite (packet-owned scenarios for device guard/backend transitions)

## Result

- packet checks: `28`
- pass checks: `20`
- non-pass checks: `8`
- packet allowlisted drifts: `8`
- packet blocking drifts: `0`

All non-pass checks are allowlisted hardened mode-split policy outcomes using:
- drift ID: `dispatch.composite_backend_fallback`
- policy source: `artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json` (`FT-P2C-007`)

Allowlisted scenario set:
- `dispatch_key/hardened:autograd_without_cpu_fail_closed` (`dispatch.composite_backend_fallback`)
- `dispatch_key/hardened:composite_route_mode_split` (`dispatch.composite_backend_fallback`)
- `dispatch_key/hardened:device_mismatch_fail_closed` (`dispatch.composite_backend_fallback`)
- `dispatch_key/hardened:dtype_mismatch_fail_closed` (`dispatch.composite_backend_fallback`)
- `dispatch_key/hardened:empty_keyset_fail_closed` (`dispatch.composite_backend_fallback`)
- `dispatch_key/hardened:no_backend_key_fail_closed` (`dispatch.composite_backend_fallback`)
- `dispatch_key/hardened:no_type_key_fail_closed` (`dispatch.composite_backend_fallback`)
- `dispatch_key/hardened:unknown_dispatch_key_fail_closed` (`dispatch.composite_backend_fallback`)

## Metamorphic and Adversarial Coverage

Metamorphic comparator coverage (`metamorphic_commutative_local`) executed in strict+hardened:
- `dispatch_key/hardened:strict_autograd_route` (`hardened`)
- `dispatch_key/hardened:strict_cpu_route` (`hardened`)
- `dispatch_key/strict:strict_autograd_route` (`strict`)
- `dispatch_key/strict:strict_cpu_route` (`strict`)

Adversarial fail-closed comparators executed in strict+hardened:
- `dispatch_key/hardened:adversarial_autograd_without_cpu` (`adversarial_autograd_without_cpu_rejected`)
- `dispatch_key/hardened:adversarial_unknown_key` (`adversarial_unknown_key_rejected`)
- `dispatch_key/strict:adversarial_autograd_without_cpu` (`adversarial_autograd_without_cpu_rejected`)
- `dispatch_key/strict:adversarial_unknown_key` (`adversarial_unknown_key_rejected`)

## Unit and E2E Traceability

Unit/property evidence anchors:
- `artifacts/phase2c/FT-P2C-007/unit_property_quality_report_v1.json`
- `crates/ft-device/src/lib.rs`
- `crates/ft-dispatch/src/lib.rs`

Differential checks reconcile against packet-007 unit/property invariants:
- device mismatch fail-closed (`DEVICE-COMPAT-007`)
- dtype mismatch fail-closed (`BACKEND-COMPAT-006`)
- autograd/backend compatibility (`BACKEND-KEYSET-004`)
- strict/hardened mode split (`BACKEND-MODE-003`)

E2E hook ownership is preserved for `bd-3v0.18.7`:
- all differential scenario IDs above map to replay/forensics scenario families under `dispatch_key/*`
- structured replay fields are enforced by packet log contract and carried into e2e forensics artifacts

## Risk/Compatibility Linkage

Threat and contract alignment:
- `artifacts/phase2c/FT-P2C-007/threat_model.md`
- `artifacts/phase2c/FT-P2C-007/contract_table.md`

Drift/allowlist decision update:
- `artifacts/phase2c/HARDENED_DEVIATION_ALLOWLIST_V1.json` now includes packet entry `FT-P2C-007` for `dispatch.composite_backend_fallback`

Residual risk remains explicit:
- non-CPU backend expansion is still deferred (`GAP-DISPATCH-007-BACKEND-DOMAIN`)
- no blocking drifts were observed in this packet differential slice
