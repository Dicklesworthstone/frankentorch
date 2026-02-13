# FT-P2C-006 — Contract Table

## Serialization Contract (Scoped)

| Contract ID | Input | Output | Error | Invariant |
|---|---|---|---|---|
| `SERDE-CHKPT-001` | checkpoint entries + mode | canonical JSON envelope | none | entries are normalized by `node_id` before hashing |
| `SERDE-CHKPT-002` | strict decode input | `CheckpointEnvelope` | `InvalidJson`, `UnknownField`, `VersionMismatch`, `ChecksumMismatch` | strict mode fail-closes unknown/incompatible payloads |
| `SERDE-CHKPT-003` | hardened decode input | `CheckpointEnvelope` + bounded diagnostics on failure | same as strict + `IncompatiblePayload` | hardened mode remains fail-closed for incompatible schema |
| `SERDE-CHKPT-004` | envelope payload | deterministic source hash | `ChecksumMismatch` | hash computed over schema, mode, normalized entries |
| `SERDE-CHKPT-005` | parity payload | `RaptorQSidecar` + `DecodeProofArtifact` | `RaptorQFailure` | sidecar includes repair manifest and decode proof hash |

## Strict vs Hardened Policy

| Mode | Behavior |
|---|---|
| strict | deny unknown fields and reject any checksum/version drift |
| hardened | include bounded diagnostic context for malformed payloads; still reject incompatible content |

## Durability Contract (RaptorQ)

1. Sidecar emits deterministic symbolization metadata (`k`, symbol size, repair count, seed).
2. Decode proof includes deterministic content hash from asupersync proof artifact.
3. Recovery path must reconstruct original checkpoint bytes exactly.
