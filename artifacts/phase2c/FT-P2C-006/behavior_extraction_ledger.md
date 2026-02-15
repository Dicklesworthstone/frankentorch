# FT-P2C-006 — Behavior Extraction Ledger

Packet: Serialization/checkpoint contract  
Legacy anchor map: `artifacts/phase2c/FT-P2C-006/legacy_anchor_map.md`

## Behavior Families (Nominal, Edge, Adversarial)

| Behavior ID | Path class | Legacy anchor family | Strict expectation | Hardened expectation | Candidate unit/property assertions | E2E scenario seed(s) |
|---|---|---|---|---|---|---|
| `FTP2C006-B01` | nominal | `doWrite`, `THPStorage_writeFileRaw`, byte-sized payload path | checkpoint envelope is canonicalized by `node_id` and source hash is deterministic for equal logical payloads | same semantic contract | `ft_serialize::checkpoint_round_trip_strict_works`, `ft_conformance::strict_serialization_conformance_is_green` | `serialization/strict:checkpoint_basic`=`7596209695846718624`, `serialization/hardened:checkpoint_basic`=`1907934922088109760` |
| `FTP2C006-B02` | nominal | `THPStorage_readFileRaw`, `doRead(file, data, storage->nbytes())` | decode preserves typed `(node_id, value, grad)` rows with sparse/`null` grad fidelity | same | `ft_serialize::legacy_snapshot_wrappers_round_trip`, `ft_conformance::hardened_serialization_conformance_is_green` | `serialization/strict:checkpoint_sparse_grad`=`13849157011772189882`, `serialization/hardened:checkpoint_sparse_grad`=`4781980514316948847` |
| `FTP2C006-B03` | edge (fail-closed parsing) | exact-size read contract (`doRead`) + storage-size compatibility assertion | unknown-field payloads fail closed (`UnknownField`) with no silent coercion | unknown-field payloads still fail closed | `ft_serialize::strict_unknown_field_fail_closed` | `serialization/strict:checkpoint_basic`=`7596209695846718624` (baseline), adversarial fixture expansion owned by `bd-3v0.17.6` |
| `FTP2C006-B04` | edge (mode split diagnostics) | strict raw read/write gate + scoped parser boundary | malformed JSON rejects with strict parser error contract | malformed JSON rejects with bounded diagnostic context (`InvalidJson`/`IncompatiblePayload`) while preserving fail-closed semantics | `ft_serialize::hardened_malformed_payload_returns_bounded_diagnostic` | `serialization/hardened:checkpoint_sparse_grad`=`4781980514316948847` (baseline), malformed corpus scenario expansion owned by `bd-3v0.17.7` |
| `FTP2C006-B05` | edge (compatibility gate) | `_storage_nbytes == nbytes` compatibility-check family | schema/version/hash drift is rejected deterministically (`VersionMismatch`, `ChecksumMismatch`) | same rejection posture; no compatibility shims | `ft_serialize::version_mismatch_is_fail_closed`, `ft_serialize::checksum_mismatch_is_fail_closed` | `serialization/strict:checkpoint_basic`=`7596209695846718624` (baseline), drift-injection scenarios owned by `bd-3v0.17.6` |
| `FTP2C006-B06` | adversarial durability | byte-exact storage reconstruction lineage + read completeness | sidecar/proof pipeline must recover original bytes exactly, and decode proof must be deterministic for identical payloads | same | `ft_serialize::sidecar_generation_and_decode_proof_are_available`, `ft_serialize::decode_proof_hash_is_deterministic` | `serialization/strict:checkpoint_basic`=`7596209695846718624`, `serialization/hardened:checkpoint_basic`=`1907934922088109760` |
| `FTP2C006-B07` | deferred parity edge | legacy archive breadth outside scalar checkpoint scope | unsupported `.pt` archive breadth (multi-storage alias graph / broad binary format) remains explicit out-of-scope and must not be silently claimed | same | deferred to packet-level closure sequence (`FT-P2C-006` downstream tasks) | candidate seeds reserved for future expansion under `bd-3v0.17.7` |

## Logging Field Expectations by Behavior Family

Mandatory deterministic replay fields (all serialization behavior families):
- `suite_id`
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `env_fingerprint`
- `artifact_refs`
- `replay_command`
- `outcome`
- `reason_code`

Serialization/durability additions:
- `source_hash`
- `schema_version`
- `repair_symbol_count`
- `constraints_symbol_count`
- `proof_hash_hex`
- `recovered_bytes`

Anchors:
- `crates/ft-serialize/src/lib.rs`
- `crates/ft-conformance/src/lib.rs`
- `artifacts/phase2c/UNIT_E2E_LOGGING_CROSSWALK_V1.json`
- `artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl`

## N/A Cross-Cutting Validation Note

This ledger is docs/planning only for packet subtask A (`bd-3v0.17.1`).
Execution evidence ownership is carried by downstream packet beads:
- contract/invariant closure: `bd-3v0.17.2`
- security/compatibility threat model: `bd-3v0.17.3`
- implementation boundaries: `bd-3v0.17.4`
- unit/property + structured logging: `bd-3v0.17.5`
- differential/metamorphic/adversarial validation: `bd-3v0.17.6`
- e2e scripts + replay/forensics logging: `bd-3v0.17.7`
- optimization/isomorphism proof: `bd-3v0.17.8`
- final evidence pack + RaptorQ closure: `bd-3v0.17.9`
