# FT-P2C-006 — Security + Compatibility Threat Model

Packet: Serialization/checkpoint contract  
Scope owner bead: `bd-3v0.17.3`

## Boundary and Assets

In-scope asset surfaces:
- checkpoint envelope encode/decode (`schema_version`, `mode`, `entries`, `source_hash`)
- strict/hardened decode behavior split
- deterministic checksum validation
- RaptorQ sidecar + decode-proof generation and integrity checks
- parity/evidence artifacts under `artifacts/phase2c/FT-P2C-006/`

Out-of-scope (explicit, non-silent):
- full PyTorch `.pt` archive breadth
- multi-storage alias graph fidelity
- large tensor binary compatibility beyond current scalar/scoped fixtures

## Threat Classes and Policy Response

| Threat ID | Abuse Class | Entry Vector | Strict Response | Hardened Response | Evidence/Test Hooks | Deterministic Scenario Seed(s) |
|---|---|---|---|---|---|---|
| `T006-01` | Unknown-field compatibility confusion | payload injects extra top-level key(s) | fail closed (`UnknownField`) | fail closed (`UnknownField`) | `ft_serialize::strict_unknown_field_fail_closed`; downstream adversarial fixture expansion in `bd-3v0.17.6` | `serialization/strict:unknown_field_injection`=`det64:76155bbfa37b082a`, `serialization/hardened:unknown_field_injection`=`det64:70d4479f246dd589` |
| `T006-02` | Schema/version downgrade or upgrade mismatch | payload `schema_version` drift | fail closed (`VersionMismatch`) | fail closed (`VersionMismatch`) | `ft_serialize::version_mismatch_is_fail_closed`; differential drift gate in packet F | `serialization/strict:version_mismatch`=`det64:f354a26a82fc0bca`, `serialization/hardened:version_mismatch`=`det64:8f0d08e0b5832e97` |
| `T006-03` | Checksum tamper / replay inconsistency | payload `source_hash` mutated | fail closed (`ChecksumMismatch`) | fail closed (`ChecksumMismatch`) | `ft_serialize::checksum_mismatch_is_fail_closed`; parity-gate enforcement | `serialization/strict:checksum_tamper`=`det64:4cb2add11028b997`, `serialization/hardened:checksum_tamper`=`det64:1fc4cf1ab4368b1c` |
| `T006-04` | Malformed payload with diagnostic abuse | malformed JSON / non-object payload | reject malformed payload | reject malformed payload with bounded diagnostic context only | `ft_serialize::hardened_malformed_payload_returns_bounded_diagnostic`; allowlist ID `serialization.bounded_malformed_diagnostic` | `serialization/strict:malformed_json`=`det64:05002972a50aa053`, `serialization/hardened:malformed_json`=`det64:b093ea04cbeb4893` |
| `T006-05` | Durability decode ambiguity | corrupted sidecar/proof or non-recoverable symbol set | decode mismatch is terminal `RaptorQFailure` | same | `ft_serialize::sidecar_generation_and_decode_proof_are_available`, `ft_serialize::decode_proof_hash_is_deterministic`; RaptorQ scrub/decode-event artifacts | `serialization/strict:raptorq_corruption_probe`=`det64:91e9a2b483885e4e`, `serialization/hardened:raptorq_corruption_probe`=`det64:8d90b3150cfe3dc4` |
| `T006-06` | Scope confusion / unsupported archive breadth | callers assume full `.pt` compatibility | explicit non-support (no silent acceptance) | explicit non-support (no silent acceptance) | packet risk note + contract-table deferred gap marker (`GAP-SERDE-001`) | planned expansion via packet-G scenarios (`bd-3v0.17.7`) |

## Compatibility Envelope

Strict mode:
- maximize scoped compatibility
- fail closed on unknown/incompatible inputs
- no behavior-altering recovery

Hardened mode:
- preserve strict external contract for acceptance/rejection
- allow only pre-allowlisted bounded defensive behavior
- current packet allowlist: `serialization.bounded_malformed_diagnostic`
- any non-allowlisted deviation is release-blocking

## Adversarial Fixture + E2E Plan

Current implemented adversarial coverage:
- unknown-field fail-closed (`ft_serialize::strict_unknown_field_fail_closed`)
- version mismatch fail-closed (`ft_serialize::version_mismatch_is_fail_closed`)
- checksum mismatch fail-closed (`ft_serialize::checksum_mismatch_is_fail_closed`)
- hardened malformed diagnostic bounding (`ft_serialize::hardened_malformed_payload_returns_bounded_diagnostic`)

Execution-bead ownership for closure evidence:
- unit/property hardening: `bd-3v0.17.5`
- differential/metamorphic/adversarial: `bd-3v0.17.6`
- e2e failure-injection scenarios + replay forensics: `bd-3v0.17.7`

## Mandatory Forensic Logging on Threat Hits

Required fields:
- `scenario_id`
- `packet_id`
- `mode`
- `seed`
- `reason_code`
- `artifact_refs`
- `replay_command`
- `env_fingerprint`

Serialization-specific additions:
- `schema_version`
- `source_hash`
- `proof_hash_hex` (durability paths)
- `recovered_bytes` (durability paths)

## Release-Gate Implications

1. Non-allowlisted hardened recovery behavior is a release-blocking failure.
2. Unknown/incompatible feature paths must fail closed in both modes.
3. Durability recoveries must emit deterministic decode-proof evidence.
4. Packet cannot be marked fully closed without explicit adversarial fixture + e2e threat coverage evidence.
