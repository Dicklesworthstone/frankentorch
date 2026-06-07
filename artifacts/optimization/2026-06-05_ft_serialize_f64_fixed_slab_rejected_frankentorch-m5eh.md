# ft-serialize F64 Fixed-Slab Native Save Rejection - frankentorch-m5eh

Date: 2026-06-05
Agent: BoldOx
Crate: ft-serialize
Target: `native_state_dict/save_single_f64_1m`
Verdict: REJECTED

## Profile Target

Fresh RCH Criterion baseline on worker `ts1` showed the F64 native save row was
the largest remaining serializer row:

```text
native_state_dict/decode_many_small_f64_1024x4 [347.28 us 360.26 us 375.44 us]
native_state_dict/save_single_f32_1m           [369.66 us 377.28 us 385.94 us]
native_state_dict/save_single_f64_1m           [936.27 us 944.80 us 953.59 us]
native_state_dict/save_single_f16_1m           [181.74 us 185.00 us 188.81 us]
native_state_dict/save_single_bf16_1m          [182.28 us 185.05 us 187.63 us]
```

Prior rejected variants had already ruled out a heap `Vec` chunk buffer and
native `BufWriter` capacity right-sizing. This pass tested a different batched
I/O primitive: a heap-free fixed slab filled in F64 value order, then written in
larger chunks.

## Candidate

Temporary candidate only:

- Baseline: per-value `write_native_bytes(writer, &value.to_le_bytes(), io_path)`.
- Candidate: fixed-size local slab, filled by `f64::to_le_bytes()` in value
  order, with one writer call per filled slab.

No retained source hunk remains after this rejection.

## Rebench

Baseline via RCH Criterion on `ts1`:

```text
native_state_dict/save_single_f64_1m
[936.27 us 944.80 us 953.59 us]
```

Candidate via RCH Criterion on the same worker:

```text
native_state_dict/save_single_f64_1m
[1.1625 ms 1.1809 ms 1.2024 ms]
```

Delta: `944.80 us -> 1.1809 ms` median, 25.0% slower.

Score: 0.0. The source hunk was restored.

## Isomorphism Proof For Candidate

- Ordering: slab fill walked the same `values` slice in the same order.
- Tie-breaking: not applicable; no comparisons.
- Floating point: no arithmetic or conversion was introduced; each value still
  used `f64::to_le_bytes()`.
- RNG: not applicable.
- Error behavior: the retained tree is unchanged. The candidate preserved
  `io_err` mapping, but batching changed writer-call granularity for arbitrary
  `Write`, which is another reason not to keep it after a regression.
- Golden output: `native_format_f64_save_bulk_golden_summary_matches_fixture`
  passed for the candidate, and the retained fixture SHA remains
  `106e45f354d304aba0ce939665820c20b312916d0ae193275a7e329ee3ce046e`.

## Proof Commands

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict --warm-up-time 1 --measurement-time 5 --sample-size 20
rch exec -- cargo fmt -p ft-serialize --check
rch exec -- cargo check -p ft-serialize --all-targets
rch exec -- cargo test -p ft-serialize native_format_f64_save_bulk_golden_summary_matches_fixture
rch exec -- cargo clippy -p ft-serialize --all-targets -- -D warnings
rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f64_1m
sha256sum artifacts/optimization/golden_outputs/ft_serialize_f64_save_bulk_frankentorch-kgs4-37.txt
```

## Next Primitive

Do not continue F64 write-buffer micro-tuning. The next pass should attack a
structurally different serializer primitive, such as a validated native-save
plan that removes duplicate layout/materialization work while preserving
pre-write fail-closed behavior and exact byte streams.
