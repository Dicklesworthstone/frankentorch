# ft-serialize Half Native Save SWAR Byte Packing - frankentorch-8t05

Date: 2026-06-05
Agent: BoldOx
Crate: ft-serialize
Target: `native_state_dict/save_single_f16_1m`, `native_state_dict/save_single_bf16_1m`
Verdict: KEPT

## Profile Target

Fresh crate-scoped RCH Criterion profile after closing `frankentorch-9w0q`
showed the native half-precision save lanes were the slowest serializer save
paths:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict --warm-up-time 1 --measurement-time 5 --sample-size 20

worker: ts2

native_state_dict/decode_many_small_f64_1024x4
time: [497.41 us 499.46 us 501.41 us]

native_state_dict/save_single_f32_1m
time: [1.3792 ms 1.4053 ms 1.4354 ms]

native_state_dict/save_single_f64_1m
time: [1.4605 ms 1.4623 ms 1.4646 ms]

native_state_dict/save_single_f16_1m
time: [1.6249 ms 1.6291 ms 1.6332 ms]

native_state_dict/save_single_bf16_1m
time: [1.6617 ms 1.6666 ms 1.6732 ms]
```

Alien primitive applied: SWAR-style byte production. The old f16/bf16 writers
appended two bytes per value. The kept lever packs four 16-bit payload bit
patterns into one little-endian `u64` lane before appending to the staging
buffer.

## One Lever

Replace the f16/bf16 per-value `extend_from_slice(&value.to_le_bytes())` loops
with a shared safe-Rust `write_native_u16_payload_values` helper that:

- keeps the same 64 KiB staging chunk size and write calls,
- reads each half value's exact `to_bits()` payload,
- packs four `u16` payloads as `v0 | v1 << 16 | v2 << 32 | v3 << 48`,
- appends `packed.to_le_bytes()`,
- emits any tail values with the same `u16::to_le_bytes()` path.

No header, key, shape, dtype tag, storage-bound, writer, f32, f64, decode, RNG,
tie-breaking, or floating-point arithmetic code changed.

## Matched Baseline

Baseline used the committed pre-lever source restored temporarily for this
measurement. RCH selected the same `ts1` worker later used by the candidate.

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single --warm-up-time 1 --measurement-time 5 --sample-size 20

worker: ts1

native_state_dict/save_single_f16_1m
time: [915.67 us 930.52 us 948.43 us]

native_state_dict/save_single_bf16_1m
time: [922.75 us 940.35 us 963.33 us]
```

## Rebench

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single --warm-up-time 1 --measurement-time 5 --sample-size 20

worker: ts1

native_state_dict/save_single_f16_1m
time: [176.65 us 181.00 us 185.48 us]

native_state_dict/save_single_bf16_1m
time: [177.58 us 182.88 us 189.00 us]
```

`RCH_WORKER=ts2` was not honored by the worker selector, so the keep decision is
based on the later same-worker `ts1` baseline above.

Delta:

- f16: `930.52 us -> 181.00 us` median, 5.14x faster.
- bf16: `940.35 us -> 182.88 us` median, 5.14x faster.

Score: Impact 3.0 x Confidence 3.0 / Effort 1.0 = 9.0, kept.

## Isomorphism Proof

- Ordering: `write_state_dict_to_writer` still iterates the same `BTreeMap`.
- Headers: magic, version, tensor count, key bytes, shape dimensions, and dtype
  tags are untouched.
- Payload bits: `Float16::to_bits` / `BFloat16::to_bits` expose the same
  underlying payload used by `to_le_bytes`; packing preserves value order as
  `[v0 low, v0 high, v1 low, v1 high, ...]`.
- Endianness: `packed.to_le_bytes()` emits the same little-endian byte order as
  four consecutive `u16::to_le_bytes()` calls. Tail values still use
  `u16::to_le_bytes()`.
- Floating point: no arithmetic or conversion is performed; only stored payload
  bits are copied.
- RNG/tie-breaking: none in this path.
- Failure behavior: validation, storage-bound checks, writer errors, and dtype
  mismatch errors are unchanged.

Golden output SHA:

```text
3b24425fcc3a77aa35d9df1ae3ebc48ba9f642124a489296b633e501336ff9fc  artifacts/optimization/golden_outputs/ft_serialize_f16_bf16_save_chunk_buffer_frankentorch-kgs4-43.txt
```

## Proof Commands

```text
cargo fmt -p ft-serialize --check
git diff --check -- crates/ft-serialize/src/lib.rs artifacts/optimization/golden_checksums.txt .beads/issues.jsonl
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize native_format_f16_bf16_save_bulk_golden_summary_matches_fixture -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-serialize --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-serialize --all-targets -- -D warnings
```

