# FT-P2C-003 — Security + Compatibility Threat Model

Packet scope: Op schema ingestion  
Subtasks: `bd-3v0.14.3` (threat model), `bd-3v0.14.6` (differential/metamorphic/adversarial execution), `bd-3v0.14.7` (packet e2e + replay/forensics), `bd-3v0.14.8` (optimization + isomorphism proof)

## 1) Threat Model Scope

Protected invariants:
- deterministic schema parse/canonicalization for scoped operator rows
- strict/hardened mode split that never alters accepted schema semantics
- fail-closed handling for malformed schema text and incompatible dispatch metadata
- deterministic forensic replay envelope for schema-ingestion incidents

Primary attack surfaces:
- malformed schema strings (token ambiguity, invalid overload names, malformed return signatures)
- operator-name normalization drift (inplace/dunder/overload canonicalization mismatch)
- out/structured alias annotation mismatch between schema rows and runtime expectations
- schema dispatch metadata mismatch causing unsafe or incorrect kernel selection

## 2) Compatibility Envelope and Mode-Split Fail-Closed Rules

| Boundary | Strict mode | Hardened mode | Fail-closed rule |
|---|---|---|---|
| schema parse (`parseSchemaOrName`) | reject malformed or ambiguous schema strings | same semantic parse contract; bounded diagnostic context only | no best-effort schema repair |
| operator identity normalization (`BaseOperatorName`/`OperatorName`) | canonical normalization required; invalid names rejected | same | ambiguous identity is terminal |
| out/structured alias metadata | explicit alias/out semantics required | same | missing or inconsistent alias metadata rejected |
| dispatch metadata mapping | unknown/incompatible schema dispatch metadata rejected | same | no fallback to implicit backend assumptions |
| replay log envelope fields | required | required | missing replay fields are reliability-gate violations |

## 3) Packet-Specific Abuse Classes and Defensive Controls

| Threat ID | Abuse class | Attack surface | Impact if unmitigated | Defensive control | Strict/Hardened expectation | Unit/property fixture mapping | Failure-injection e2e scenario seed(s) |
|---|---|---|---|---|---|---|---|
| `THR-301` | malformed schema token stream | `parseSchemaOrName` grammar path | schema accepted with wrong structure or overload identity | deterministic parser rejection + explicit error taxonomy | strict=fail, hardened=fail | `ft_dispatch::schema_parser_rejects_malformed_tokens`, `ft_dispatch::schema_parser_rejects_illegal_overload_name` | `op_schema/strict:malformed_schema_rejected`=`15378956306914137809`, `op_schema/hardened:malformed_schema_rejected`=`13132209225168612872` |
| `THR-302` | operator-name normalization drift | `BaseOperatorName`/`OperatorName` parse and unambiguous naming | kernel binding mismatches across overload/inplace variants | canonical normalization + round-trip identity checks | strict=deterministic, hardened=deterministic | `ft_dispatch::base_operator_name_parse_inplace_suffix_contract`, `ft_dispatch::operator_unambiguous_name_stable` | `op_schema/strict:operator_name_normalization`=`13467211897931571851`, `op_schema/hardened:operator_name_normalization`=`13933643675815328326` |
| `THR-303` | out/structured alias metadata corruption | out schema rows and structured delegate metadata | unsafe mutability/alias handling and parity drift | explicit out-alias invariants + structured delegate preservation checks | strict=fail on mismatch, hardened=fail on mismatch | `ft_dispatch::schema_out_variant_requires_mutable_out_alias`, `ft_dispatch::structured_delegate_ref_is_preserved` | `op_schema/strict:add_out_schema_alignment`=`73962521145086383`, `op_schema/hardened:add_out_schema_alignment`=`4600075214105047437` |
| `THR-304` | dispatch metadata incompatibility | schema `dispatch:` mapping into scoped keyset routing | unsafe or wrong kernel route selection | reject unknown/incompatible metadata before route resolution | strict=fail, hardened=fail | `ft_dispatch::schema_dispatch_keyset_rejects_unknown_backend_key`, `ft_dispatch::schema_dispatch_keyset_requires_cpu_backend_for_scoped_ops` | `op_schema/strict:dispatch_metadata_incompatible`=`5876961574866201650`, `op_schema/hardened:dispatch_metadata_incompatible`=`3598587240931457689` |
| `THR-305` | replay evidence omission/tampering | structured forensics payload | unreplayable schema incidents and audit loss | required deterministic log contract + reliability gate | same in both modes | conformance logging contract checks (`StructuredCaseLog`) | full-suite e2e/reliability gate runs |
| `THR-306` | symbolic-shape parity ambiguity | schema rows requiring unresolved symbolic-shape semantics | silent acceptance of undefined schema behavior | explicit gap marker + deferred closure dependency | strict=fail-closed on unknown symbolic-shape semantics, hardened=same semantics with bounded diagnostics | `symbolic_shape_gap_marker` fixture gated by `GAP-SCHEMA-001` | `op_schema/strict:symbolic_shape_gap_marker`, `op_schema/hardened:symbolic_shape_gap_marker` |

## 4) Mandatory Forensic Logging + Replay Artifacts for Incidents

For every schema security/compat incident, logs must include:
- `suite_id`
- `scenario_id`
- `packet_id`
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

Required artifact linkage chain:
1. packet e2e log entry (`artifacts/phase2c/e2e_forensics/ft-p2c-003.jsonl`)
2. packet failure triage (`artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_003_v1.json`)
3. packet failure index envelope (`artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_003_v1.json`)
4. packet replay/forensics linkage (`artifacts/phase2c/FT-P2C-003/e2e_replay_forensics_linkage_v1.json`)
5. packet replay/forensics narrative (`artifacts/phase2c/FT-P2C-003/e2e_replay_forensics_linkage_v1.md`)

## 5) Residual Risks and Deferred Controls

Residual risks:
- this packet only scopes a constrained operator family; full native operator universe ingestion is deferred.
- symbolic-shape schema parity remains explicit implementation debt tracked by `GAP-SCHEMA-001`, with active fail-closed marker coverage under `frankentorch-795j`.

Deferred controls and ownership:
- `bd-3v0.14.6` status: completed.
  - differential source report: `artifacts/phase2c/conformance/differential_report_v1.json`
  - packet slice: `artifacts/phase2c/FT-P2C-003/differential_packet_report_v1.json`
  - reconciliation note: `artifacts/phase2c/FT-P2C-003/differential_reconciliation_v1.md`
  - drift posture: `38` checks, `0` non-pass, `0` packet allowlisted drift, `0` packet blocking drift
- `bd-3v0.14.7` status: completed.
  - packet e2e log: `artifacts/phase2c/e2e_forensics/ft-p2c-003.jsonl`
  - packet triage: `artifacts/phase2c/e2e_forensics/crash_triage_ft_p2c_003_v1.json`
  - packet failure index: `artifacts/phase2c/e2e_forensics/failure_forensics_index_ft_p2c_003_v1.json`
  - packet linkage pair: `artifacts/phase2c/FT-P2C-003/e2e_replay_forensics_linkage_v1.{json,md}`
  - run posture: `10` entries (`strict=5`, `hardened=5`), `0` failed entries, `0` triaged incidents
- `bd-3v0.14.8` status: completed.
  - optimization delta: `artifacts/phase2c/FT-P2C-003/optimization_delta_v1.json`
  - isomorphism note: `artifacts/phase2c/FT-P2C-003/optimization_isomorphism_v1.md`
  - optimization lever: packet-filtered e2e suite gating (`e2e-packet-scope-suite-gating`)
  - latency-tail delta: `p50 -82.686%`, `p95 -85.172%`, `p99 -85.172%`, `mean -83.518%`
- `bd-3v0.14.9` status: completed.
  - parity gate: `artifacts/phase2c/FT-P2C-003/parity_gate.yaml`
  - parity report: `artifacts/phase2c/FT-P2C-003/parity_report.json`
  - packet sidecars: `artifacts/phase2c/FT-P2C-003/parity_report.raptorq.json`, `artifacts/phase2c/FT-P2C-003/parity_report.decode_proof.json`
- extend non-CPU dispatch/schema families under later backend-expansion packets (`FT-P2C-007` chain).

## 6) N/A Cross-Cutting Validation Note

This risk note is docs/planning for packet subtask C, with execution-status cross-links for closure visibility.
Execution evidence ownership:
- `bd-3v0.14.5` (unit/property + structured logs)
- `bd-3v0.14.6` (differential/metamorphic/adversarial)
- `bd-3v0.14.7` (e2e/replay forensics)
