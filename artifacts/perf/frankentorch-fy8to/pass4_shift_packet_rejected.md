# frankentorch-fy8to Pass 4 Shift-Packet Rejection

Date: 2026-06-12

Scope: one-lever source attempt for bead `frankentorch-fy8to`.

## Lever Tried

Private AED-derived Schur-window shift packet for the `eigvals` path only:

- copy a bounded active Hessenberg tail window,
- compute a local Schur/eigenvalue packet on the copy,
- validate finite values, Hessenberg shape, exceptional-shift/max-total gates,
  and scalar bulge-start normalization,
- feed only the resulting `x/y/w` shift packet into the existing scalar
  single-bulge Francis chase.

Full `eig` remained on the scalar path.

## Baseline

Fresh same-session remote baseline:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigvals_f64_256x256
worker: vmi1227854
eigvals_f64_256x256 time: [24.909 ms 25.230 ms 25.565 ms]
```

Historical pass-1 strict golden SHA remains:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

## Behavior Gate

Focused eig tests passed remotely:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu --lib eig -- --nocapture
worker: ovh-a
result: 21 passed; 0 failed
```

Strict golden failed on `vmi1227854`:

```text
expected n=256 eigvals_digest=0x00b87b4996340204
candidate n=256 eigvals_digest=0xdaa6738e0f31a016
candidate n=256 eig_digest=0x00b87b4996340204
```

The full-`eig` fallback remained bit-identical for the printed digest, but the
`eigvals` stream order/bit pattern changed. That violates the pass-3 strict
golden contract, so the source hunk was rejected before any after-benchmark
could qualify as a keep result.

## Evidence

```text
5626ad02333c1c8400cf736452bfa3b59eadd6578b9f6eeec267e5a899c1d9b5  pass4_before_eigvals_f64_256x256.log
2c55203f82c11ea212dcc98ba793037558cb0ad4857979b89718de626ea2fc4b  pass4_candidate_test_eig.log
120f3a4ddf46aacc6283dab770c647cce04948ccc4d5ea8d7f4141946b39ed7b  pass4_candidate_eigvals_golden.stdout
```

Final source state:

```text
git diff --name-only -- crates/ft-kernel-cpu
<empty>
```

## Verdict

Rejected. Score `0.0`: behavior gate failed.

Next route: a deeper standalone Schur-window kernel/AED artifact that can prove
strict fallback before changing the global eigenvalue stream. Do not repeat
range/index cuts, diagnostic-only helpers, or this shift-packet-only source
shape.
