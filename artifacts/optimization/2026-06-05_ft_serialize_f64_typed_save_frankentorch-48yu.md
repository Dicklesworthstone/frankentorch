# ft-serialize F64 Native Save Buffer Right-Sizing Rejection - frankentorch-48yu

Date: 2026-06-05
Agent: BoldOx
Crate: ft-serialize
Target: `native_state_dict/save_single_f64_1m`
Verdict: REJECTED

## Profile Target

This pass followed the post-`frankentorch-6x8x` RCH reprofile on worker `ts1`.
After f32 SWAR packing, the remaining native serializer rows were:

- `native_state_dict/decode_many_small_f64_1024x4`: `[316.92 us 321.03 us 325.50 us]`
- `native_state_dict/save_single_f32_1m`: `[341.76 us 346.86 us 351.07 us]`
- `native_state_dict/save_single_f64_1m`: `[911.24 us 927.46 us 941.74 us]`
- `native_state_dict/save_single_f16_1m`: `[178.58 us 182.19 us 185.97 us]`
- `native_state_dict/save_single_bf16_1m`: `[175.60 us 177.68 us 180.18 us]`

The f64 native save path was the largest serializer row.

## Candidate

The tested lever was a safe-Rust batched-I/O primitive inspired by the alien-graveyard
I/O batching guidance: right-size the native `BufWriter` from the encoded state_dict
capacity, with an 8 MiB cap, so the f64 1M payload could avoid repeated forced
flushes. The candidate promoted `native_state_dict_encoded_capacity` out of test-only
code and used it to choose the writer buffer capacity after validation.

This was not the same as the earlier rejected `frankentorch-j125` 64 KiB f64 chunk
buffer mirror: no f64 payload chunk buffer was added, and the value loop remained
the existing direct per-value writer path.

## Isomorphism Proof For Candidate

- Ordering stayed unchanged: `BTreeMap` iteration still determined tensor order.
- Header, key bytes, shape bytes, dtype tags, value traversal, and error mapping were unchanged.
- f64 payload generation was untouched; each value still used `f64::to_le_bytes()`.
- Floating-point arithmetic, RNG, and tie-breaking were unaffected.
- The only changed state was the private `BufWriter` capacity; the byte stream and flush-at-end contract were intended to remain identical.

## Rebench

Baseline via RCH Criterion on `ts1`:

```text
native_state_dict/save_single_f64_1m
[911.24 us 927.46 us 941.74 us]
```

Candidate via RCH Criterion on the same worker:

```text
native_state_dict/save_single_f64_1m
[1.7172 ms 1.7259 ms 1.7369 ms]
```

Delta: `927.46 us -> 1.7259 ms` median, 1.86x slower.

Score: 0.0. The source hunk was removed.

## Golden Output

The source was restored before commit, so no serializer behavior changed. The relevant
f64 golden fixture remains:

```text
106e45f354d304aba0ce939665820c20b312916d0ae193275a7e329ee3ce046e  artifacts/optimization/golden_outputs/ft_serialize_f64_save_bulk_frankentorch-kgs4-37.txt
```

## Next Primitive

The regression indicates the f64 bottleneck is not primarily forced flush count in this
benchmark. The next f64 pass should attack a different primitive: a byte-production
pipeline that avoids both per-value `Write` dispatch and a second payload copy, or a
validated native-save plan that removes redundant metadata/layout work while preserving
pre-write fail-closed behavior. Target ratio: at least 2x on `save_single_f64_1m`.
