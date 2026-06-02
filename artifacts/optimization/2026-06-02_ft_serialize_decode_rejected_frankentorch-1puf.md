# ft-serialize Native Decode Rejected Pass

- Bead: `frankentorch-1puf`
- Skills: `/profiling-software-performance`, `/extreme-software-optimization`, `/alien-graveyard`
- Crate: `ft-serialize`
- Target benchmark: `native_state_dict/decode_many_small_f64_1024x4`

## Profile Target

Scenario: native state-dict decode for 1024 small f64 tensors, width 4.

Baseline:

```text
worker: vmi1293453
command: rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
native_state_dict/decode_many_small_f64_1024x4: [342.17 us 345.59 us 349.52 us]
```

Profiler note:

```text
command: rch exec -- perf stat -e cycles,instructions,cache-misses cargo bench -p ft-serialize --bench serialize_bench -- --warm-up-time 1 --measurement-time 3 --sample-size 10
result: blocked by worker perf_event_paranoid=4
```

Ranked hotspot evidence:

| Rank | Location | Metric | Evidence |
|------|----------|--------|----------|
| 1 | `load_state_dict_from_bytes` -> `read_f64_payload` | repeated fixed-width decode for 1024 tensors | Criterion scenario and source-path inspection |
| 2 | native key parsing + `BTreeMap` insertion | one key/order check per tensor | source-path inspection |

## Alien Recommendation Card

Candidate: replace the f64 payload reader's manual `Vec::with_capacity` +
`push` loop with an exact-size iterator `collect`, preserving fixed-width
little-endian construction.

Mapped primitives:

- Constants-kill-you: fixed-width decoding overhead dominates this tiny-payload
  benchmark, so a micro-lever must prove it beats iterator and allocation
  constants.
- Vectorized/morsel execution: rejected for this pass because tensor payloads
  are only 4 f64 values each; there is no useful chunk size without changing the
  native format or batching across records.

Fallback: keep the existing manual push loop.

## Candidate Result

After candidate:

```text
worker: vmi1227854
command: rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
native_state_dict/decode_many_small_f64_1024x4: [394.21 us 411.56 us 435.70 us]
```

Decision: rejected. The run was on a different worker and the candidate was
slower than the original baseline, so it has no valid >=2.0 score and no code
change ships.

## Isomorphism

The rejected candidate preserved ordering, duplicate-key fail-closed behavior,
dtype/error classes, RNG independence, and f64 bit construction through
`f64::from_le_bytes`. Because the benchmark did not prove a win, the source was
restored to the original loop and no golden output was added.
