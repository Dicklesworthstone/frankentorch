# ft-serialize half native-save zero-copy proof

Date: 2026-06-06
Agent: VioletForest
Scope: `ft-serialize`

## Target selection

After the f32/f64 save keeps, the serializer save panel still showed half
precision writes in the hundreds of microseconds. A matched baseline was taken
with the committed SWAR/chunk half writer restored temporarily:

```text
ts1 native_state_dict/save_single_f16_1m:  [202.76 us 216.60 us 238.92 us]
ts1 native_state_dict/save_single_bf16_1m: [208.82 us 215.50 us 224.58 us]
```

## Lever

Enable the `half` crate's `bytemuck` feature and write contiguous `f16`/`bf16`
payloads as borrowed byte views on little-endian targets:

```text
bytemuck::cast_slice::<Float16, u8>(values)
bytemuck::cast_slice::<BFloat16, u8>(values)
```

The existing SWAR/chunk writer remains compiled for non-little-endian targets.

## Re-benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single --warm-up-time 1 --measurement-time 5 --sample-size 20
```

RCH selected `ts1`, matching the baseline worker.

Criterion after:

```text
ts1 native_state_dict/save_single_f16_1m:  [6.2930 us 6.4110 us 6.5065 us]
ts1 native_state_dict/save_single_bf16_1m: [6.0789 us 6.1574 us 6.2529 us]
```

Delta:

```text
f16:  216.60 us -> 6.4110 us, about 33.8x faster
bf16: 215.50 us -> 6.1574 us, about 35.0x faster
```

Score: Impact 5 * Confidence 5 / Effort 2 = 12.5, keep.

## Isomorphism proof

Ordering: state dict entry ordering remains the existing `BTreeMap` order. The
header, key, shape, stride, dtype tag, and payload ordering are unchanged.

Tie-breaking: no comparator or tie-breaking path changed.

Floating point: no arithmetic or conversion is performed. The `half` crate marks
`f16` and `bf16` as `#[repr(transparent)]` over `u16` with `Pod` when its
`bytemuck` feature is enabled. On little-endian targets, the byte view writes the
same low-byte/high-byte order as the previous `to_bits().to_le_bytes()` path.
The non-little-endian fallback still uses the previous `to_bits` SWAR writer.

RNG: no RNG state or sampling path is involved.

Golden output:

```text
artifacts/optimization/golden_outputs/ft_serialize_f16_bf16_save_chunk_buffer_frankentorch-kgs4-43.txt: OK
```

## Verification

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize native_format_f16_bf16_save_bulk_golden_summary_matches_fixture -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-serialize --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-serialize --all-targets -- -D warnings
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize
cargo fmt -p ft-core -p ft-serialize --check
ubs crates/ft-core/Cargo.toml crates/ft-serialize/src/lib.rs
git diff --check -- Cargo.lock crates/ft-core/Cargo.toml crates/ft-serialize/src/lib.rs
```

All listed gates passed. UBS exited 0; remaining scanner output is file-wide
warning inventory.

## Re-profile

Same-worker post-keep full panel:

```text
ts1 native_state_dict/decode_many_small_f64_1024x4: [334.98 us 339.78 us 344.32 us]
ts1 native_state_dict/save_single_f32_1m:           [5.7066 us 5.7672 us 5.8484 us]
ts1 native_state_dict/save_single_f64_1m:           [5.7299 us 5.8139 us 5.8770 us]
ts1 native_state_dict/save_single_f16_1m:           [5.7529 us 5.8141 us 5.8849 us]
ts1 native_state_dict/save_single_bf16_1m:          [5.9701 us 6.1693 us 6.4792 us]
```

Next target: native-state decode is now dominant. The next attempt should be a
structural many-small-tensor decode primitive, not another scalar f64 byte loop.
