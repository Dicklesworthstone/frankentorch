# FT-P2C-003 — Contract Table + Strict/Hardened Invariant Spec

Packet: Op schema ingestion  
Dependencies: `bd-3v0.14.1` behavior extraction ledger + packet anchor map

## Machine-Checkable Contract Row Schema

Each contract row is complete only if it defines:
- preconditions
- postconditions
- invariant class ID(s)
- strict-mode semantics
- hardened-mode semantics
- fail-closed boundary decision
- unit/property mapping
- differential/metamorphic/adversarial intent
- e2e scenario ID mapping
- drift posture (`forbidden`, `allowlisted_hardened_only`, `deferred_with_gap_id`)

## Contract Rows

| Contract ID | Behavior ID | Preconditions | Postconditions | Invariant class | Strict semantics | Hardened semantics | Fail-closed boundary | Unit/property mapping | Differential/adversarial intent | E2E scenario IDs | Drift posture |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `OPSCHEMA-INGEST-001` | `FTP2C003-B01` | schema row text is sourced from canonical `native_functions.yaml` entry | typed schema record preserves op name, overload token, argument/return shape | `FT-I1`, `FT-I6` | parse and canonicalize deterministically; no implicit schema rewrites | same semantic parse contract | invalid schema row is terminal parse error | `ft_dispatch::schema_row_parse_round_trips_add_tensor_signature`, `ft_dispatch::operator_name_parse_preserves_overload_token` | compare parsed schema identity against deterministic oracle fixtures for scoped op rows | `op_schema/strict:add_tensor_schema_roundtrip`, `op_schema/hardened:add_tensor_schema_roundtrip` | forbidden |
| `OPSCHEMA-NAME-002` | `FTP2C003-B02` | operator symbol may include inplace/dunder/overload variants | normalized operator identity is reversible (`name`, `overload`) and unambiguous | `FT-I1`, `FT-I6` | canonical normalization required | same | ambiguous/illegal names rejected (no auto-repair) | `ft_dispatch::base_operator_name_parse_inplace_suffix_contract`, `ft_dispatch::operator_unambiguous_name_stable` | verify schema-name normalization parity and unambiguous-name stability | `op_schema/strict:operator_name_normalization`, `op_schema/hardened:operator_name_normalization` | forbidden |
| `OPSCHEMA-OUT-003` | `FTP2C003-B03` | out/structured variant metadata is present | out-alias semantics and structured delegate metadata remain explicit and machine-checkable | `FT-I2`, `FT-I6` | mutable/out alias relationships must be explicit | same | missing out-alias metadata is contract violation | `ft_dispatch::schema_out_variant_requires_mutable_out_alias`, `ft_dispatch::structured_delegate_ref_is_preserved` | differential checks validate out-variant schema tags and alias annotations | `op_schema/strict:add_out_schema_alignment`, `op_schema/hardened:add_out_schema_alignment` | forbidden |
| `OPSCHEMA-PARSER-004` | `FTP2C003-B04` | parser input may be full schema string or name-only | parser classification preserves declaration type (`OperatorName` vs `FunctionSchema`) | `FT-I5`, `FT-I6` | wrong form is rejected with deterministic failure reason | same parsing semantics; diagnostics can include bounded context | no fallback from malformed full-schema into name-only acceptance | `ft_dispatch::parse_schema_or_name_classifies_name_only_vs_full_schema`, `ft_dispatch::schema_parser_rejects_invalid_arrow_forms` | adversarial differential cases stress separator/arrow grammar boundaries | `op_schema/strict:parser_name_vs_schema_split`, `op_schema/hardened:parser_name_vs_schema_split` | allowlisted_hardened_only (`op_schema.diagnostic_context_only`) |
| `OPSCHEMA-MALFORMED-005` | `FTP2C003-B05` | schema tokens may be malformed/ambiguous | malformed schema is rejected with deterministic error taxonomy | `FT-I5` | malformed syntax fails closed | same fail-closed behavior | never infer a "best effort" schema from invalid text | `ft_dispatch::schema_parser_rejects_malformed_tokens`, `ft_dispatch::schema_parser_rejects_illegal_overload_name` | adversarial malformed corpus must produce deterministic pass/fail evidence | `op_schema/strict:malformed_schema_rejected`, `op_schema/hardened:malformed_schema_rejected` | forbidden |
| `OPSCHEMA-DISPATCH-006` | `FTP2C003-B06` | schema dispatch metadata declares backend key routing for scoped ops | incompatible/unknown schema dispatch metadata is rejected before kernel selection | `FT-I3`, `FT-I6` | incompatible schema-to-keyset mapping fails closed | same fail-closed behavior | unknown backend keys or incompatible keysets are terminal | `ft_dispatch::schema_dispatch_keyset_rejects_unknown_backend_key`, `ft_dispatch::schema_dispatch_keyset_requires_cpu_backend_for_scoped_ops` | adversarial checks validate parser+dispatch integration against hostile metadata | `op_schema/strict:dispatch_metadata_incompatible`, `op_schema/hardened:dispatch_metadata_incompatible` | forbidden |
| `OPSCHEMA-SYMSHAPE-007` | `FTP2C003-B07` | symbolic-shape schema surface appears in scoped parse domain | behavior remains explicitly gated until symbolic-shape closure is implemented | `FT-I6` | unknown symbolic-shape semantics cannot pass silently | same with bounded diagnostics | symbolic-shape unresolved cases must be raised as explicit gap markers | `symbolic_shape_gap_marker` fixture in `op_schema_cases.json` | differential/e2e checks assert the explicit `GAP-SCHEMA-001` marker and fail-closed dispatch posture until closure | `op_schema/strict:symbolic_shape_gap_marker`, `op_schema/hardened:symbolic_shape_gap_marker` | deferred_with_gap_id (`GAP-SCHEMA-001`) |

## Contract Violation Logging Requirements

Every op-schema contract violation event must include:
- `event_type` (contract ID + invariant class)
- `scenario_id`
- `mode`
- `seed`
- `op_name`
- `overload_name`
- `schema_digest`
- `dispatch_keyset_bits`
- `reason_code`
- `artifact_refs`
- `replay_command`
- `env_fingerprint`

Anchors:
- `crates/ft-conformance/src/logging.rs:8`
- `crates/ft-conformance/src/lib.rs:300`
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`

## N/A Cross-Cutting Validation Note

This artifact update is docs/planning only for packet subtask B.
Execution evidence is deferred to:
- `bd-3v0.14.5` (unit/property)
- `bd-3v0.14.6` (differential/metamorphic/adversarial)
- `bd-3v0.14.7` (e2e/logging)
