# ft-data RandomSampler direct-output shuffle rejection

Bead: `frankentorch-1m8j`
Date: 2026-06-05
Agent: BlackThrush
Crate: `ft-data`
Target: `sampler/without_replacement_repeated_passes_4096x256`
Verdict: rejected and source hunk restored

## Profile-backed target

The ready perf queue was ownership/policy blocked, so this pass used the
profile-backed fallback target named by the previous weighted-sampler rejection:
`RandomSampler::indices` repeated no-replacement passes.

Fresh baseline via rch Criterion:

```text
worker: ts2
command: RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/without_replacement_repeated_passes_4096x256 --warm-up-time 1 --measurement-time 5 --sample-size 20
sampler/without_replacement_repeated_passes_4096x256: [8.0759 ms 8.1927 ms 8.2835 ms]
```

## Candidate

Replace the scratch `idx` reset/copy/extend path with a direct output-buffer
primitive:

- append `0..size` directly into the result buffer for each pass;
- Fisher-Yates shuffle that just-appended output slice in place;
- for the final remainder, append a full identity pass, shuffle it, then
  truncate back to the requested prefix length.

The candidate preserved the same shuffle length and RNG draw count for every
full pass and the remainder pass.

## Proof while candidate was present

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-data random_sampler -- --nocapture
```

Result:

```text
worker: ts2
29 sampler tests passed, including random_sampler_repeated_passes_preserve_exact_order
and random_sampler_repeated_passes_match_refill_reference_order.
```

Existing exact-sequence golden remained:

```text
827bc5a601bacca033c8127d6a1efe79711987bcaf41c789e6889166db100a52  artifacts/optimization/golden_outputs/random_sampler_pass16.txt
```

## Rebenchmark

Same-worker candidate run:

```text
worker: ts2
command: RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo bench -p ft-data --bench sampler_bench -- sampler/without_replacement_repeated_passes_4096x256 --warm-up-time 1 --measurement-time 5 --sample-size 20
sampler/without_replacement_repeated_passes_4096x256: [8.0035 ms 8.0756 ms 8.1323 ms]
```

Delta:

```text
median: 8.1927 ms -> 8.0756 ms
speedup: 1.0145x
score: 1.0145 * 0.95 / 1.0 = 0.96
```

## Decision

Rejected below the required Score >= 2.0 keep gate. The source hunk was restored.

Next deeper target: stop repeated-pass memory-copy tuning for this sampler. The
next `ft-data` pass should attack a structurally different data-pipeline
primitive, such as zero-copy batch views for dataloader collation or a new
sampler contract where distribution parity, not bit-for-bit sample order, is
acceptable.
