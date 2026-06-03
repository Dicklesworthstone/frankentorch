# ft-serialize F32 Save Chunk-Buffer Probe

Bead: `frankentorch-snnd`
Agent: TurquoisePine
Outcome: rejected; no runtime change kept.

## Target

Fallback profile-backed target after `br ready --json` returned no ready work and
`br list --status=open --json` returned no open beads.

Benchmark:

```text
native_state_dict/save_single_f32_1m
```

Fresh pre-change rch baseline:

```text
worker: vmi1156319
command: CARGO_TARGET_DIR=target/turquoise-pine-ft-serialize-f32-chunk-baseline rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f32_1m --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [1.1711 ms 1.2290 ms 1.2764 ms]
```

Same-worker old-code evidence from the retained F32 bulk-write pass:

```text
worker: vmi1293453
command: CARGO_TARGET_DIR=target/turquoise-pine-ft-serialize-hwsk-after-confirm rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f32_1m --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [717.19 us 727.73 us 740.41 us]
criterion mean: 0.723594 ms
```

## Tested Lever

One lever only: change `FT_NATIVE_F32_VALUE_CHUNK_BYTES` from `64 * 1024` to
the existing `FT_NATIVE_SAVE_BUFFER_BYTES` value of 1 MiB, so the f32 scratch
buffer chunk size matches the outer native-save `BufWriter` capacity.

## Behavior Proof

The attempted lever did not change observable encoding semantics:

- Ordering: `BTreeMap` key order and per-value order stayed unchanged.
- Tie-breaking: N/A.
- Floating point: no arithmetic changed; every f32 value still used
  `to_le_bytes`.
- RNG: N/A.
- Error behavior: validation and unsupported dtype handling stayed before the
  f32 write branch.
- Golden bytes: the attempted bead-specific fixture matched existing F32 native
  save bytes with sha256
  `a4f99bf82139749e11ea6a626324f0fb77a7498f797085350280c5d63fabc233`.

## Rebenchmark

After via rch:

```text
worker: vmi1293453
command: CARGO_TARGET_DIR=target/turquoise-pine-ft-serialize-f32-chunk-after rch exec -- cargo bench -p ft-serialize --bench serialize_bench -- native_state_dict/save_single_f32_1m --warm-up-time 1 --measurement-time 5 --sample-size 20
time: [1.0924 ms 1.1211 ms 1.1444 ms]
```

The after-run was not comparable to the fresh `vmi1156319` baseline, but it was
directly comparable to the prior old-code `vmi1293453` run. That same-worker
comparison regressed from `727.73 us` to `1.1211 ms`, about 54.0 percent slower.

## Decision

Rejected. Score is below 2.0 because the same-worker profile evidence regressed.
The source, checksum, and test changes were removed; no runtime change is kept.
