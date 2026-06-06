# ft-serialize f32 native-save zero-copy proof

Date: 2026-06-06
Agent: VioletForest
Scope: `ft-serialize`

## Target selection

`br ready --json` still returned no ready `[perf]` beads after the f64 keep.
The post-keep serializer profile on `vmi1227854` made
`native_state_dict/save_single_f32_1m` the next save-path hotspot:

```text
native_state_dict/save_single_f32_1m: [295.92 us 310.96 us 326.28 us]
```

This pass does not revisit the earlier rejected per-value writer or chunk-size
tuning. It replaces the current SWAR/chunk staging with a different primitive:
a borrowed little-endian byte view over the contiguous `f32` payload.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `vmi1227854`

Criterion baseline from the post-f64 full panel:

```text
native_state_dict/save_single_f32_1m: [295.92 us 310.96 us 326.28 us]
```

## Lever

`DType::F32` native save now calls `bytemuck::cast_slice::<f32, u8>` on
little-endian targets and writes that borrowed byte view directly. The obsolete
f32 chunk-size constant was removed because the f32 path no longer stages a
temporary payload buffer. Non-little-endian targets keep scalar `to_le_bytes`
serialization.

## Re-benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f32_1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `vmi1227854`

Criterion after:

```text
native_state_dict/save_single_f32_1m: [1.4495 us 1.4874 us 1.5280 us]
```

Delta: `310.96 us -> 1.4874 us` p50, about `209x` faster.

Score: Impact 5 * Confidence 5 / Effort 1 = 25.0, keep.

## Isomorphism proof

Ordering: state dict entry ordering remains the existing `BTreeMap` order. The
header, key, shape, stride, dtype tag, and payload ordering are unchanged.

Tie-breaking: no comparator or tie-breaking path changed.

Floating point: no arithmetic is performed. The little-endian target path writes
the exact in-memory IEEE-754 `f32` bit patterns as bytes, which is identical to
`to_le_bytes` on little-endian targets, preserving NaN payloads, infinities, and
signed zero. The non-little-endian fallback still writes `to_le_bytes`.

RNG: no RNG state or sampling path is involved.

Golden output:

```text
artifacts/optimization/golden_outputs/ft_serialize_f32_save_bulk_pass26.txt: OK
```

## Verification

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize native_format_f32_save_bulk_golden_summary_matches_fixture -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-serialize --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-serialize --all-targets -- -D warnings
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize
cargo fmt -p ft-serialize --check
ubs crates/ft-serialize/src/lib.rs
git diff --check -- crates/ft-serialize/src/lib.rs
```

All listed gates passed.

## Re-profile

Same-worker post-keep full panel:

```text
native_state_dict/decode_many_small_f64_1024x4: [344.04 us 357.44 us 369.17 us]
native_state_dict/save_single_f32_1m:          [1.3625 us 1.4135 us 1.4626 us]
native_state_dict/save_single_f64_1m:          [1.5273 us 1.5720 us 1.6140 us]
native_state_dict/save_single_f16_1m:          [158.59 us 167.59 us 174.69 us]
native_state_dict/save_single_bf16_1m:         [144.15 us 149.15 us 156.11 us]
```

Next target: native-state decode is now dominant in the serializer panel.
