# frankentorch-gxpb2 pass 5 branch-hoist rejection

Bead: `frankentorch-gxpb2`
Date: 2026-06-13
Worker: `vmi1227854`

## Profile-backed target

The live target remains the non-symmetric Francis QR floor in `eigvals_f64_256x256`.
Prior `fql10`/`npxbw`/`gxpb2` evidence shows the serial one-bulge Francis sweep is
the shared geev/eigvals wall after blocked Hessenberg and Schur-vector work.

## Lever Tried

The candidate split the production Francis bulge row and column updates into
separate `notlast` and final-bulge paths:

- row update: remove the per-element `if notlast` test inside `for j in k..row_end`;
- column update: remove the per-element `if notlast` test inside `for i in 0..=jmax`;
- keep the same row-before-column order and the same arithmetic expression order
  for every written slot.

This is not the full `gxpb2` AED/multishift dispatch. It was a bounded hot-loop
candidate against the currently profiled Francis QR floor.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 \
  rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench \
  eigvals_f64_256x256 -- --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Actual worker selected by RCH: `vmi1227854`.

Baseline row:

```text
eigvals_f64_256x256 time: [23.270 ms 24.057 ms 24.717 ms]
```

## Candidate Proof

Focused eig tests:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 \
  rch exec -- cargo test -j 1 -p ft-kernel-cpu --lib eig -- --nocapture
```

Result: `24 passed; 0 failed` on `vmi1227854`.

Strict golden:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 \
  rch exec -- cargo run -j 1 -p ft-kernel-cpu --release --example eigvals_golden
```

Digest lines:

```text
frankentorch-l9xod eigvals_golden n=64
eigvals_digest=0xbc0583d464b1a211
eig_digest=0xbc0583d464b1a211
frankentorch-l9xod eigvals_golden n=128
eigvals_digest=0x763c4b15d92c4b89
eig_digest=0x763c4b15d92c4b89
frankentorch-l9xod eigvals_golden n=256
eigvals_digest=0x00b87b4996340204
eig_digest=0x00b87b4996340204
```

Extracted strict stdout SHA-256:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

Isomorphism proof:

- Ordering: unchanged bottom-up `en` loop, active-window search, `m` search,
  row update before column update, and deflation order.
- Tie-breaking: unchanged `eps * s` split tests, exceptional shift cadence
  (`its == 10 || its == 20`), and `max_total` fallback policy.
- Floating point: each slot computes `p2`, then writes the same rows/columns in
  the same expression order as the baseline; only the invariant branch location
  moved.
- Complex slots: unchanged 1x1/2x2 eigenvalue slot writes and conjugate-pair
  convention.
- RNG: none.

## Candidate Benchmark

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 \
  rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench \
  eigvals_f64_256x256 -- --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Actual worker selected by RCH: `vmi1227854`.

Candidate row:

```text
eigvals_f64_256x256 time: [23.456 ms 23.911 ms 24.581 ms]
```

The median ratio is `24.057 / 23.911 = 1.006x`, but the Criterion intervals
overlap heavily. This does not clear the real-win requirement.

## Verdict

Rejected. Score: `0.0`.

The source hunk was removed after the benchmark, and `crates/ft-kernel-cpu/src/lib.rs`
has no final diff. Scoped formatting passed after removal:

```bash
cargo fmt -p ft-kernel-cpu --check
```

## Next Route

Do not repeat branch/range/const micro-levers on `gxpb2`. The next pass should
attack a structurally different primitive:

- strict scalar-shift operation tape that proves identical shift samples,
  selected `m`, deflation counters, Schur bits, and eigenvalue slots, then
  batches only proven-independent far updates; or
- a bounded Schur-window/AED record with explicit shift-list and deflation proof
  before any public dispatch.
