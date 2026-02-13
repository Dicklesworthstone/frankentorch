# Phase-2C Artifact Schema Lock v1

This document freezes the machine-checkable packet artifact topology for `FT-P2C-*` packets.

## Required Files Per Packet

Each packet directory `artifacts/phase2c/FT-P2C-00X/` MUST include:

1. `legacy_anchor_map.md`
2. `contract_table.md`
3. `fixture_manifest.json`
4. `parity_gate.yaml`
5. `risk_note.md`
6. `parity_report.json`
7. `parity_report.raptorq.json`
8. `parity_report.decode_proof.json`

Missing any file => packet status `NOT_READY`.

## Mandatory JSON Fields

### `fixture_manifest.json`
- `packet_id`
- `fixtures`
- `modes`
- `status`

### `parity_report.json`
- `packet_id`
- `suite`
- `strict`
- `hardened`
- `status`
- `generated_from`

### `parity_report.raptorq.json`
- `artifact_id`
- `artifact_type`
- `source_hash`
- `raptorq`
- `scrub`

### `parity_report.decode_proof.json`
- `artifact_id` with packet prefix
- either `decode_proof` OR `decode_events`

Missing mandatory field => packet status `NOT_READY`.

## Mandatory `parity_gate.yaml` Sections

- `packet_id: FT-P2C-00X` (exact packet match)
- `strict:`
- `hardened:`
- `checks:`
- `artifacts:`

Missing section => packet status `NOT_READY`.

## Validator Command

Run:

```bash
cargo run -p ft-conformance --bin validate_phase2c_artifacts
```

Exit code policy:
- `0`: all packets `READY`
- `1`: at least one packet `NOT_READY`
- `2`: invalid root/path error
