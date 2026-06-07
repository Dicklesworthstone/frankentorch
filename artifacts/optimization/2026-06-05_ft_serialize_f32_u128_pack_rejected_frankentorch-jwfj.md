# ft-serialize F32 U128 Native Save Pack Rejection - frankentorch-jwfj

Date: 2026-06-05
Agent: BoldOx
Crate: ft-serialize
Target: `native_state_dict/save_single_f32_1m`
Verdict: REJECTED

## Profile Target

After two F64 native-save variants regressed, a fresh RCH Criterion profile on
worker `ts2` showed the top non-F64 serializer row was F32 native save:

```text
native_state_dict/decode_many_small_f64_1024x4 [485.64 us 486.52 us 487.82 us]
native_state_dict/save_single_f32_1m           [537.81 us 539.46 us 541.18 us]
native_state_dict/save_single_f64_1m           [1.4140 ms 1.4218 ms 1.4303 ms]
native_state_dict/save_single_f16_1m           [270.76 us 271.18 us 271.89 us]
native_state_dict/save_single_bf16_1m          [270.97 us 272.44 us 273.99 us]
```

## Candidate

Temporary candidate only:

- Baseline: current F32 SWAR writer packs two `f32::to_bits()` payloads into one
  little-endian `u64` append.
- Candidate: widen packing to four `f32::to_bits()` payloads into one
  little-endian `u128` append, preserving the same 64 KiB chunking and byte
  stream.

The implementation was restored after same-worker rebenching regressed.

## Rebench

Supplied baseline via RCH Criterion on `ts2`:

```text
native_state_dict/save_single_f32_1m
[537.81 us 539.46 us 541.18 us]
```

Same-session pre-edit baseline on `ts2`:

```text
native_state_dict/save_single_f32_1m
[534.42 us 534.93 us 535.41 us]
```

Candidate on `ts2`:

```text
native_state_dict/save_single_f32_1m
[537.53 us 539.57 us 542.15 us]
```

Delta versus same-session baseline: `534.93 us -> 539.57 us` median, 0.87%
slower.

Score: 0.0. The source hunk was restored.

## Isomorphism Proof For Candidate

- Ordering: `BTreeMap` iteration and tensor value iteration were unchanged.
- Headers: key bytes, shape bytes, dtype tag, and tensor count were untouched.
- Floating point: no arithmetic was introduced; each payload used `f32::to_bits()`.
- Endianness: the candidate packed values into low-to-high 32-bit fields and
  emitted `u128::to_le_bytes()`, preserving value byte order.
- Tail handling: 1, 2, or 3 remaining values emitted the same byte sequence.
- RNG and tie-breaking: not applicable.
- Golden output: `native_format_f32_save_bulk_golden_summary_matches_fixture`
  passed; checksum remained
  `a4f99bf82139749e11ea6a626324f0fb77a7498f797085350280c5d63fabc233`.

## Proof Commands

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f32_1m --warm-up-time 1 --measurement-time 5 --sample-size 20
rch exec -- cargo fmt -p ft-serialize --check
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-serialize --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize native_format_f32_save_bulk_golden_summary_matches_fixture
sha256sum artifacts/optimization/golden_outputs/ft_serialize_f32_save_bulk_pass26.txt
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-serialize --all-targets -- -D warnings
```

## Next Primitive

Stop serializer byte-emission micro-levers. The next pass should profile a
different crate or a structurally different algorithmic surface.
