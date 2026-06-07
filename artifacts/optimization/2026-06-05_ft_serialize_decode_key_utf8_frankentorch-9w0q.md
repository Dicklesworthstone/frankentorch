# ft-serialize Native Decode Key UTF-8 Fast Path - frankentorch-9w0q

Date: 2026-06-05
Agent: BoldOx
Crate: ft-serialize
Target: `native_state_dict/decode_many_small_f64_1024x4`
Verdict: REJECTED

## Profile Target

`load_state_dict_from_bytes` decoded each native tensor key by copying the key bytes
into a temporary `Vec<u8>` and then validating/owning that vector with
`String::from_utf8`. The benchmark payload has 1024 small tensors, so this looked
like an avoidable per-key byte-vector allocation before the final owned key needed
by the `BTreeMap`.

Alien primitive tested: zero-copy parser validation. Validate the source byte slice
first, then allocate the final owned key exactly once.

## One Lever

Temporary candidate only:

- Baseline: `String::from_utf8(data[pos..key_end].to_vec())`.
- Candidate: `std::str::from_utf8(&data[pos..key_end])?.to_owned()`.

No value parsing, dtype dispatch, shape parsing, ordering, duplicate-key handling,
RNG, tie-breaking, or floating-point arithmetic was changed during the trial. The
candidate source hunk was removed after the matched rebench regressed.

## Baseline

Initial profile baseline on `ts2`:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20

native_state_dict/decode_many_small_f64_1024x4
time: [479.28 us 481.37 us 483.73 us]
```

Matched baseline on `ts1`, with the committed copying implementation restored:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20

native_state_dict/decode_many_small_f64_1024x4
time: [312.72 us 317.59 us 323.96 us]
```

## Rebench

Candidate result on the same `ts1` worker:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/decode_many_small_f64_1024x4 --warm-up-time 1 --measurement-time 5 --sample-size 20

native_state_dict/decode_many_small_f64_1024x4
time: [323.43 us 327.40 us 331.30 us]
```

Delta: `317.59 us -> 327.40 us` median on `ts1`, about 3.1% slower.

Score: Impact 0.0 x Confidence 3.0 / Effort 1.0 = 0.0, rejected.

## Isomorphism Proof

- Ordering: candidate left `BTreeMap` insertion and duplicate-key detection unchanged.
- Key bytes: valid UTF-8 keys would still produce the same owned `String` values.
- Failure behavior: invalid UTF-8 mapped to the same `TensorIOError::Corrupt` reason.
- Shape/dtype/value parsing was untouched.
- Floating point: F64 payload bytes remained decoded by the same `read_f64_payload`
  path, so no arithmetic or bit interpretation changed.
- RNG/tie-breaking: none in this path.

Golden SHA proof was not needed for the final commit because no source hunk was kept.
The final tree preserves the pre-pass native decode implementation.

## Next Target

Do not continue key-parser micro-tuning. Re-profile `ft-serialize` and attack a
larger serializer primitive next: slab-oriented typed byte emission for native save
paths, producing exact native byte streams in larger reusable slabs without
per-value writer calls or temporary per-key byte vectors.
