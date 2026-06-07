# ft-serialize f64 native-save zero-copy proof

Date: 2026-06-06
Agent: VioletForest
Scope: `ft-serialize`

## Target selection

`br ready --json` had no ready `[perf]` beads. The in-flight linalg bead
`frankentorch-rd1s` was already claimed and its files were exclusively reserved
by BlackThrush, so this pass used the profiler-evident serializer hotspot
instead of colliding.

Prior serializer f64-save attempts already rejected heap chunk buffers, fixed
slabs, and BufWriter right-sizing. This pass used a different primitive:
write the contiguous `f64` payload as a little-endian byte slice on little-endian
targets, with a scalar `to_le_bytes` fallback on big-endian targets.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `vmi1227854`

Criterion baseline:

```text
native_state_dict/save_single_f64_1m: [647.45 us 688.60 us 733.37 us]
```

## Lever

`DType::F64` native save now calls `bytemuck::cast_slice::<f64, u8>` on
little-endian targets and writes that borrowed byte view directly. No temporary
payload buffer is allocated and no per-element byte conversion loop runs on the
normal target.

## Re-benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f64_1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `vmi1227854`

Criterion after:

```text
native_state_dict/save_single_f64_1m: [1.3870 us 1.4258 us 1.4553 us]
```

Delta: `688.60 us -> 1.4258 us` p50, about `483x` faster.

Score: Impact 5 * Confidence 5 / Effort 1 = 25.0, keep.

## Isomorphism proof

Ordering: state dict entry ordering remains the existing `BTreeMap` order. The
header, key, shape, stride, dtype tag, and payload ordering are unchanged.

Tie-breaking: no comparator or tie-breaking path changed.

Floating point: no arithmetic is performed. The little-endian target path writes
the exact in-memory IEEE-754 `f64` bit patterns as bytes, which is identical to
`to_le_bytes` on little-endian targets, preserving NaN payloads, infinities, and
signed zero. The non-little-endian fallback still writes `to_le_bytes`.

RNG: no RNG state or sampling path is involved.

Golden output:

```text
artifacts/optimization/golden_outputs/ft_serialize_f64_save_bulk_frankentorch-kgs4-37.txt: OK
```

## Verification

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize native_format_f64_save_bulk_golden_summary_matches_fixture -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-serialize --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-serialize --all-targets -- -D warnings
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize
cargo fmt -p ft-serialize --check
ubs Cargo.toml crates/ft-serialize/Cargo.toml crates/ft-serialize/src/lib.rs
git diff --check -- Cargo.toml crates/ft-serialize/Cargo.toml crates/ft-serialize/src/lib.rs
```

All listed gates passed.

## Re-profile

Same-worker post-keep full panel:

```text
native_state_dict/decode_many_small_f64_1024x4: [279.39 us 294.32 us 311.65 us]
native_state_dict/save_single_f32_1m:          [295.92 us 310.96 us 326.28 us]
native_state_dict/save_single_f64_1m:          [1.5425 us 1.6054 us 1.6498 us]
native_state_dict/save_single_f16_1m:          [139.83 us 144.17 us 149.01 us]
native_state_dict/save_single_bf16_1m:         [169.64 us 185.54 us 197.06 us]
```

Next target: native-state decode or f32 save, both still hundreds of
microseconds in the crate-scoped serializer panel.
