# ft-serialize Validated Native Save Plan Rejection - frankentorch-w0ez

Date: 2026-06-05
Agent: BoldOx
Crate: ft-serialize
Target: `native_state_dict/save_single_f64_1m`
Verdict: REJECTED

## Profile Target

This pass followed the fixed-slab F64 rejection in `frankentorch-m5eh`.
The current same-worker baseline on `ts1` remained:

```text
native_state_dict/save_single_f64_1m
[936.27 us 944.80 us 953.59 us]
```

The bead hypothesis was that native save duplicated validation/materialization
work: `validate_state_dict_native_save` validates each tensor and the writer
then performs the same layout/storage slice checks while emitting bytes.

## Candidate

Temporary candidate only:

- Build a validated native-save plan after fail-closed validation.
- Store borrowed tensor payload slices in that plan.
- Write headers and payloads from the plan, avoiding the second
  `contiguous_values()` / storage-slice validation step.

The implementation was restored after the benchmark regressed.

## Rebench

Baseline via RCH Criterion on `ts1`:

```text
native_state_dict/save_single_f64_1m
[936.27 us 944.80 us 953.59 us]
```

Candidate via RCH Criterion on `ts1`:

```text
native_state_dict/save_single_f64_1m
[947.19 us 976.52 us 1.0238 ms]
```

Delta: `944.80 us -> 976.52 us` median, 3.36% slower.

Score: 0.0. The source hunk was restored.

## Isomorphism Proof For Candidate

- Ordering: candidate retained the same `BTreeMap` iteration order.
- Headers: key bytes, shape bytes, dtype tags, and tensor count were emitted in
  the same sequence.
- Floating point: values were borrowed and serialized as payload bits; no
  arithmetic, rounding, sorting, or comparison was introduced.
- RNG and tie-breaking: not applicable.
- Failure behavior: retained source is unchanged. Candidate validation still
  ran before writing, but the measured regression made this path unshippable.
- Golden output: `native_format_f64_save_bulk_golden_summary_matches_fixture`
  passed, and the fixture SHA remained
  `106e45f354d304aba0ce939665820c20b312916d0ae193275a7e329ee3ce046e`.

## Proof Commands

```text
rch exec -- cargo fmt -p ft-serialize --check
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-serialize --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize native_format_f64_save_bulk_golden_summary_matches_fixture
sha256sum artifacts/optimization/golden_outputs/ft_serialize_f64_save_bulk_frankentorch-kgs4-37.txt
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f64_1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

## Next Primitive

Do not continue F64 native-save metadata or buffering variants. Re-profile and
move to a different serializer row or a structurally different decode/tensor
construction primitive.
