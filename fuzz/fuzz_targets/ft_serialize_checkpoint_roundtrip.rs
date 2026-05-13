#![no_main]

use ft_serialize::{
    CheckpointMode, DecodeMode, SnapshotEntry, decode_checkpoint, encode_checkpoint,
};
use libfuzzer_sys::fuzz_target;

const MAX_INPUT_BYTES: usize = 4096;
const MAX_ENTRIES: usize = 64;

fuzz_target!(|data: &[u8]| {
    if data.len() < 2 || data.len() > MAX_INPUT_BYTES {
        return;
    }

    let n_entries = usize::from(1 + (data[0] % MAX_ENTRIES as u8)).min(MAX_ENTRIES);
    let mode = if data[1] & 1 == 0 {
        CheckpointMode::Strict
    } else {
        CheckpointMode::Hardened
    };
    let body = &data[2..];

    // Build SnapshotEntry list. Use moderate finite f64 values so
    // Hardened mode's NaN/inf rejection doesn't trip on encode.
    let mut entries: Vec<SnapshotEntry> = Vec::with_capacity(n_entries);
    for i in 0..n_entries {
        let value_byte = body.get(i % body.len().max(1)).copied().unwrap_or(0) as i32;
        let value = (value_byte - 128) as f64 / 13.0;
        let grad_byte = body
            .get((n_entries + i) % body.len().max(1))
            .copied()
            .unwrap_or(0);
        let grad = if grad_byte % 4 == 0 {
            None
        } else {
            let g = (grad_byte as i32 - 128) as f64 / 17.0;
            Some(g)
        };
        entries.push(SnapshotEntry {
            node_id: i,
            value,
            grad,
        });
    }

    // Encode → decode → compare. encode_checkpoint normalizes
    // (stable sort + dedup) so we expect the decoded entries to
    // match what encode produced.
    let encoded = match encode_checkpoint(&entries, mode) {
        Ok(s) => s,
        Err(_) => return,
    };

    let decode_mode = match mode {
        CheckpointMode::Strict => DecodeMode::Strict,
        CheckpointMode::Hardened => DecodeMode::Hardened,
    };
    let decoded = match decode_checkpoint(&encoded, decode_mode) {
        Ok(env) => env,
        Err(e) => panic!("decode after encode failed: {e:?}"),
    };

    // Decoded envelope must round-trip: re-encoding it produces
    // bit-identical JSON. This catches any normalization drift
    // between encode and decode.
    let re_encoded = encode_checkpoint(&decoded.entries, decoded.mode)
        .expect("re-encode of decoded entries should succeed");
    assert_eq!(
        encoded, re_encoded,
        "encode → decode → encode is not idempotent"
    );

    // Per-entry value preservation: every value and grad survives
    // the roundtrip bit-exactly (modulo serde_json's f64 → string
    // → f64 path, which is also bit-exact for finite values).
    assert_eq!(
        decoded.entries.len(),
        entries.len(),
        "round-trip entry count mismatch"
    );
    // Note: entries are normalized (stable-sorted by node_id +
    // dedup); since we pushed with monotonic node_id 0..n_entries
    // there are no duplicates and the sort is a no-op.
    for (i, (orig, got)) in entries.iter().zip(decoded.entries.iter()).enumerate() {
        assert_eq!(orig.node_id, got.node_id, "entry {i} node_id drift");
        assert_eq!(
            orig.value.to_bits(),
            got.value.to_bits(),
            "entry {i} value drift: orig {} bits, got {} bits",
            orig.value.to_bits(),
            got.value.to_bits()
        );
        assert_eq!(orig.grad.is_some(), got.grad.is_some(), "entry {i} grad presence drift");
        if let (Some(og), Some(gg)) = (orig.grad, got.grad) {
            assert_eq!(
                og.to_bits(),
                gg.to_bits(),
                "entry {i} grad drift: orig {} bits, got {} bits",
                og.to_bits(),
                gg.to_bits()
            );
        }
    }
});
