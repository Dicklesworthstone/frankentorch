# FT-P2C-006 — Contract Table + Strict/Hardened Invariant Spec

Packet: Serialization/checkpoint contract  
Dependencies: `bd-3v0.17.1` behavior extraction ledger + packet anchor map

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
| `SERDE-CHKPT-001` | `FTP2C006-B01` | checkpoint entry list is well-formed and mode is explicit (`strict`/`hardened`) | canonical envelope emitted with entries normalized by `node_id` and deterministic `source_hash` | `FT-I1`, `FT-I6` | canonicalization + hashing are deterministic and replay-stable | same semantic contract | malformed envelope construction paths are rejected, never auto-repaired | `ft_serialize::checkpoint_round_trip_strict_works`, `ft_conformance::strict_serialization_conformance_is_green` | compare local envelope/hash behavior against fixture-backed parity reports | `serialization/strict:checkpoint_basic`, `serialization/hardened:checkpoint_basic` | forbidden |
| `SERDE-CHKPT-002` | `FTP2C006-B02` | checkpoint payload contains sparse/optional gradients | decode preserves typed `(node_id, value, grad)` rows exactly, including `null` gradient fidelity | `FT-I2`, `FT-I6` | sparse-grad payload remains byte/semantic stable under decode | same | missing/coerced grad states are contract violations | `ft_serialize::legacy_snapshot_wrappers_round_trip`, `ft_conformance::hardened_serialization_conformance_is_green` | verify sparse-grad parity in fixture and differential packet checks | `serialization/strict:checkpoint_sparse_grad`, `serialization/hardened:checkpoint_sparse_grad` | forbidden |
| `SERDE-STRICT-003` | `FTP2C006-B03` | strict decode receives payload containing extra/unknown fields | decode terminates with `UnknownField` (or strict parser rejection) and no output envelope | `FT-I5` | unknown fields are terminal fail-closed errors | unknown fields remain terminal fail-closed errors | no unknown-field compatibility shim allowed | `ft_serialize::strict_unknown_field_fail_closed` | adversarial mutation corpus must preserve deterministic fail-closed reason taxonomy | `serialization/strict:checkpoint_basic` (baseline); adversarial seed expansion in `bd-3v0.17.6` | forbidden |
| `SERDE-MODE-004` | `FTP2C006-B04` | malformed JSON/incompatible top-level payload is provided | strict: parser error; hardened: bounded diagnostic payload context while remaining rejecting | `FT-I5`, `FT-I6` | malformed payload rejected without recovery | malformed payload rejected; bounded diagnostic context permitted | hardened diagnostics may not alter acceptance semantics | `ft_serialize::hardened_malformed_payload_returns_bounded_diagnostic` | differential checks assert reject-in-both-modes with bounded hardened diagnostic envelope | `serialization/hardened:checkpoint_sparse_grad` (baseline); malformed scenario expansion in `bd-3v0.17.7` | allowlisted_hardened_only (`serialization.bounded_malformed_diagnostic`) |
| `SERDE-COMPAT-005` | `FTP2C006-B05` | envelope version/hash drift injected into otherwise parseable payload | deterministic incompatibility signal emitted (`VersionMismatch` / `ChecksumMismatch`) | `FT-I5`, `FT-I6` | reject on any version/hash mismatch | same | no downgrade-to-warning path | `ft_serialize::version_mismatch_is_fail_closed`, `ft_serialize::checksum_mismatch_is_fail_closed` | adversarial drift tests must prove fail-closed boundary and deterministic reason code | `serialization/strict:checkpoint_basic` (baseline), drift injections tracked under `bd-3v0.17.6` | forbidden |
| `SERDE-RQ-006` | `FTP2C006-B06` | checkpoint payload bytes + requested repair symbols | sidecar and decode proof generated; recovered payload is byte-identical; proof hash deterministic for same input | `FT-I1`, `FT-I3` | recovery must be exact, deterministic, and reproducible | same | any decode mismatch is terminal `RaptorQFailure` | `ft_serialize::sidecar_generation_and_decode_proof_are_available`, `ft_serialize::decode_proof_hash_is_deterministic` | durability checks validate sidecar/proof and corruption-probe evidence in packet/global artifacts | `serialization/strict:checkpoint_basic`, `serialization/hardened:checkpoint_basic` | forbidden |
| `SERDE-SCOPE-007` | `FTP2C006-B07` | workload requires full PyTorch archive breadth (multi-storage alias graph, broad binary archive semantics) | explicit out-of-scope declaration retained until closure beads land | `FT-I6` | unsupported breadth cannot be silently advertised as supported | same | out-of-scope paths must remain explicit gap markers | deferred to packet closure sequence (`bd-3v0.17.3`..`bd-3v0.17.9`) | future differential/e2e hooks required before closure | reserved for future packet-006 expansion seeds in `bd-3v0.17.7` | deferred_with_gap_id (`GAP-SERDE-001`) |

## Contract Violation Logging Requirements

Every serialization contract violation event must include:
- `event_type` (contract ID + invariant class)
- `scenario_id`
- `mode`
- `seed`
- `reason_code`
- `artifact_refs`
- `replay_command`
- `env_fingerprint`
- `source_hash` (when available)
- `schema_version` (when available)

Durability-specific violation additions:
- `repair_symbol_count`
- `constraints_symbol_count`
- `proof_hash_hex`
- `recovered_bytes`

Anchors:
- `crates/ft-serialize/src/lib.rs`
- `crates/ft-conformance/src/logging.rs`
- `crates/ft-conformance/src/lib.rs`
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`

## N/A Cross-Cutting Validation Note

This artifact update is docs/planning only for packet subtask B.
Execution evidence is deferred to:
- `bd-3v0.17.5` (unit/property)
- `bd-3v0.17.6` (differential/metamorphic/adversarial)
- `bd-3v0.17.7` (e2e/logging)
