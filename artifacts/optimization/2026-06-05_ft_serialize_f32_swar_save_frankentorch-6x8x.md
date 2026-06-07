# ft-serialize F32 Native Save SWAR Byte Packing - frankentorch-6x8x

Date: 2026-06-05
Agent: BoldOx
Crate: ft-serialize
Target: `native_state_dict/save_single_f32_1m`
Verdict: KEPT

## Profile Target

This pass followed the post-`frankentorch-8t05` RCH reprofile on worker `ts1`.
After half/bfloat16 SWAR packing, the remaining native serializer hotspots were:

- `native_state_dict/decode_many_small_f64_1024x4`: `[327.59 us 332.56 us 338.48 us]`
- `native_state_dict/save_single_f32_1m`: `[885.47 us 918.07 us 964.31 us]`
- `native_state_dict/save_single_f64_1m`: `[889.74 us 903.28 us 919.12 us]`
- `native_state_dict/save_single_f16_1m`: `[178.08 us 180.47 us 182.41 us]`
- `native_state_dict/save_single_bf16_1m`: `[172.84 us 176.73 us 181.10 us]`

The profile-backed target for this bead was the f32 native save path.

## One Lever

Replace per-value f32 byte appends with a safe-Rust typed payload emitter that packs
two `f32::to_bits()` payloads into one little-endian `u64` append while preserving the
same 64 KiB staging buffer and chunk boundaries.

## Benchmark Evidence

Baseline via RCH Criterion on `ts1`:

```text
native_state_dict/save_single_f32_1m
[885.47 us 918.07 us 964.31 us]
```

Candidate via RCH Criterion on the same worker:

```text
native_state_dict/save_single_f32_1m
[352.59 us 357.15 us 361.25 us]
```

Delta: `918.07 us -> 357.15 us` median, 2.57x faster.

Score: 9.0 = Impact 3.0 x Confidence 3.0 / Effort 1.0.

## Isomorphism Proof

- Map ordering is unchanged: `BTreeMap` iteration still determines state_dict entry order.
- Header, key bytes, shape payload, dtype tags, writer path, and error behavior are unchanged.
- Floating-point arithmetic is unchanged: values are never recomputed, rounded, sorted, or compared.
- Payload bits are unchanged: each f32 still uses `to_bits()`.
- Byte order is unchanged: `packed.to_le_bytes()` emits the same little-endian stream as two consecutive `u32::to_le_bytes()` calls for the low and high lanes.
- Odd tail values use the same `u32::to_le_bytes()` path as before.
- RNG, ordering, and tie-breaking are unaffected; the save path is deterministic serialization only.
- f64, f16, bf16, and decode paths are untouched by this lever.

## Golden Output

The f32 bulk-save golden fixture remains:

```text
a4f99bf82139749e11ea6a626324f0fb77a7498f797085350280c5d63fabc233  artifacts/optimization/golden_outputs/ft_serialize_f32_save_bulk_pass26.txt
```

Verification:

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

## Proof Commands

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f32_1m --warm-up-time 1 --measurement-time 5 --sample-size 20`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize native_format_f32_save_bulk_golden_summary_matches_fixture -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-serialize --all-targets`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-serialize --all-targets -- -D warnings`
- `cargo fmt -p ft-serialize --check`
- `git diff --check -- crates/ft-serialize/src/lib.rs`
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`

## Next Profile Target

After this keep, the serializer profile should be refreshed. The likely next primitive is the f64 native save path, targeting a similar typed slab/SWAR byte emitter while preserving exact f64 bit streams and the `ft_serialize_f64_save_bulk_frankentorch-kgs4-37` golden SHA.
