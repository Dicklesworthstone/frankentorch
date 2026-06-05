# ft-serialize F32 Native Save Stream Rejection - frankentorch-ouig

Date: 2026-06-05
Agent: BoldOx
Crate: ft-serialize
Target: `native_state_dict/save_single_f32_1m`

## Profile Target

The bead targeted the F32 native save byte-production path:

- `save_state_dict` writes a single 1,000,000-element F32 tensor to `/dev/null`.
- The committed implementation builds one 64 KiB byte chunk at a time, then writes the chunk through the existing `BufWriter`.
- The candidate replaced chunk construction with per-value `to_le_bytes()` writes through the same writer.

This was a one-lever probe of whether avoiding the temporary chunk `Vec` beats the current chunked byte-production path.

## Same-Worker Criterion Evidence

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f32_1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

Committed chunked baseline:

```text
native_state_dict/save_single_f32_1m
time: [892.40 us 902.04 us 913.28 us]
```

Per-value writer candidate:

```text
native_state_dict/save_single_f32_1m
time: [988.99 us 1.0017 ms 1.0145 ms]
```

Delta: candidate is about 11.0% slower by median.

Score: 0.0, rejected below the required Score >= 2.0 gate.

## Isomorphism / Golden Evidence

The candidate only changed byte-production mechanics for already-contiguous F32 values:

- Tensor and key ordering unchanged (`BTreeMap` iteration retained).
- Shape metadata, dtype tag, and format header unchanged.
- F32 little-endian byte order unchanged (`f32::to_le_bytes()` retained per value).
- No RNG, tie-breaking, or floating-point arithmetic changes.

After restoring the committed chunked source, the golden ledger passed:

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Relevant existing F32 native-save golden remained OK:

```text
artifacts/optimization/golden_outputs/ft_serialize_f32_save_bulk_pass26.txt: OK
```

## Decision

Do not keep the per-value writer. The source hunk was removed and `write_native_f32_values` remains on the committed chunked implementation.

Next deeper primitive for this surface should not be another per-value writer tweak. The likely larger lever is a format-writer abstraction that emits typed contiguous storage in larger slabs without per-value writer calls while preserving the exact native byte stream.
