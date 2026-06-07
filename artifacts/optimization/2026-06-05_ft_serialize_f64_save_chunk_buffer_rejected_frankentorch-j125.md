# ft-serialize F64 Native Save Chunk Buffer Rejection

- Bead: `frankentorch-j125`
- Agent: `BoldOx`
- Surface: `crates/ft-serialize`
- Target row: `native_state_dict/save_single_f64_1m`
- Verdict: rejected and source hunk removed

## Profile-Backed Baseline

The pass used the disjoint ft-serialize fallback surface because the ready perf
queue was empty and the active no-gaps/kernel and optimizer beads were already
claimed by other agents.

Baseline command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts2`

Relevant baseline row:

```text
native_state_dict/save_single_f64_1m
time: [1.4206 ms 1.4231 ms 1.4275 ms]
```

## Candidate

The candidate added a `write_native_f64_values` helper mirroring the existing
F32/F16/BF16 chunk-buffer writers and routed the F64 save branch through it.

Intended primitive:

- Constants-kill-you: replace one million small buffered write calls with
  64 KiB byte chunks.
- Byte-format-preserving bulk serialization: keep the same FTSV header,
  key order, shape order, dtype tag, and little-endian value bytes.

## Isomorphism Proof

- Ordering: `BTreeMap` traversal, key bytes, shape bytes, dtype tag, and value
  traversal order were unchanged.
- Tie-breaking: not applicable; no comparisons changed.
- Floating point: no arithmetic changed. Each value still used
  `f64::to_le_bytes`, preserving NaN payloads, signed zero, and infinities.
- RNG: not applicable.
- Error behavior: validation, unsupported dtype handling, and I/O error mapping
  stayed on the existing `TensorIOError` paths.

Proof commands while the candidate was present:

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-serialize native_format_f64_save_bulk_golden_summary_matches_fixture -- --nocapture
```

Proof results:

```text
golden_checksums: all tracked outputs OK
native_format_f64_save_bulk_golden_summary_matches_fixture: 1 passed on ts2
```

## Rebench

Candidate command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f64_1m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Candidate result:

```text
worker: ts2
native_state_dict/save_single_f64_1m
time: [1.4751 ms 1.4766 ms 1.4781 ms]
```

Delta:

```text
median: 1.4231 ms -> 1.4766 ms
elapsed: 3.8% slower
score: 0.0
```

## Closeout

The source hunk was removed after the same-worker regression. The existing
scalar F64 writer remains in place.

Do not repeat this as another chunk-buffer micro-lever. The next serializer
pass should attack a deeper primitive, such as reducing validation/write double
walks, avoiding redundant per-save metadata work, or redesigning native save
around a byte-production pipeline with a stronger profile-backed target.
