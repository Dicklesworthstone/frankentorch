#![no_main]

use libfuzzer_sys::fuzz_target;

// Cap fuzz inputs at 256 KiB. The decode path itself rejects payloads
// over MAX_CHECKPOINT_PAYLOAD_BYTES (1 MiB) but the cheaper short-
// circuit here keeps the fuzzer iterating on inputs the validator
// would actually walk. Mirrors the safetensors target's cap pattern.
const MAX_CHECKPOINT_FUZZ_BYTES: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    if data.len() > MAX_CHECKPOINT_FUZZ_BYTES {
        return;
    }
    // decode_checkpoint takes &str, so we must filter non-UTF-8 first
    // — feeding raw bytes through a `from_utf8_unchecked` would itself
    // be UB and defeat the point of fuzzing.
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };

    // The decoder exercises:
    //   * MAX_CHECKPOINT_PAYLOAD_BYTES guard (1 MiB)
    //   * serde_json::from_str into CheckpointEnvelope (deny_unknown_fields
    //     in Strict via #[serde], deny-list in Hardened via key sweep)
    //   * SnapshotEntry validation (NaN/inf rejection in hardened mode,
    //     duplicate node_id detection, bounded-string truncation)
    //   * normalize_entries: stable sort by node_id, deterministic dedup
    //   * extract_unknown_field error-string parsing (strict path)
    //
    // Hardened mode walks the JSON value tree pre-deserialize to enforce
    // the allow-list, so it has a strictly larger surface than Strict.
    // Fuzz both modes so a regression in either decoder surfaces.
    let _ = ft_serialize::decode_checkpoint(text, ft_serialize::DecodeMode::Strict);
    let _ = ft_serialize::decode_checkpoint(text, ft_serialize::DecodeMode::Hardened);

    // decode_snapshot is a thin wrapper over Strict mode that returns the
    // entries list rather than the envelope. Driving it too keeps the
    // wrapper from drifting out of sync.
    let _ = ft_serialize::decode_snapshot(text);
});
