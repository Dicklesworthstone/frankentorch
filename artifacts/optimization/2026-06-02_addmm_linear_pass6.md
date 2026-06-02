# Addmm Linear Optimization Pass 6

Bead: `frankentorch-m4b7`

## Target

Profile-backed target: `linear_forward/hidden/2048` in `ft-api` Criterion benchmarks.

The prior campaign measurement bundle `tests/artifacts/perf/20260601T2325Z-rustickite/`
showed `linear_forward` hidden sizes 256/512/1024/2048 running at approximately
1.13/2.42/4.81/9.67 ms, while the PyTorch anchor was approximately
0.114/0.224/0.441/0.886 ms. That made the linear path a profile-backed fallback
target while another active agent owned the conv2d hotspot.

## Baseline

Command:

```text
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-m4b7-target rch exec -- cargo bench -p ft-api --bench ops_bench -- linear_forward/hidden/2048 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts2`

Criterion result:

```text
linear_forward/hidden/2048 time: [21.514 ms 21.735 ms 21.912 ms]
throughput: [2.9908 Melem/s 3.0152 Melem/s 3.0462 Melem/s]
```

## Lever

Attempted one lever in `ft-kernel-cpu`: remove the second full-size addmm output
allocation by running GEMM directly into the returned output buffer, then applying
`beta * input + alpha * gemm` in place.

The attempted expression order was kept equivalent to the previous collect path:

```text
beta * input + alpha * gemm_value
```

## Behavior Proof

Focused rch checks while the lever was applied:

```text
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-m4b7-target rch exec -- cargo test -p ft-kernel-cpu addmm_tensor_contiguous -- --nocapture
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-m4b7-target rch exec -- cargo test -p ft-api functional_linear_with_bias -- --nocapture
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-m4b7-target rch exec -- cargo test -p ft-api session_tensor_addmm -- --nocapture
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-m4b7-target rch exec -- cargo check -p ft-kernel-cpu --all-targets
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-m4b7-target rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings
```

All commands exited 0.

Golden output:

```text
functional_linear_with_bias values: [11.0, 21.0]
session_tensor_addmm values: [14.0, 25.0, 20.0, 31.0]
session_tensor_addmm_scaled values: [17.0, 22.0, 27.0, 32.0]
```

Golden sha256:

```text
f333da438688311750d1b2c7be14fd0fb63bc0b6e350e2e0dd0c0b5dfb1236c0  artifacts/optimization/golden_outputs/addmm_linear_pass6.txt
```

Isomorphism notes:

- Ordering and tie-breaking: addmm output iteration order and broadcast indexing unchanged.
- Floating point: attempted lever preserved `beta * input + alpha * gemm_value` expression order.
- RNG: not involved.
- Golden-output checksum: verified with `sha256sum -c artifacts/optimization/golden_checksums.txt`.

## Re-benchmark

Command:

```text
CARGO_TARGET_DIR=/data/tmp/frankentorch-rustickite-m4b7-target rch exec -- cargo bench -p ft-api --bench ops_bench -- linear_forward/hidden/2048 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts2`

Criterion result:

```text
linear_forward/hidden/2048 time: [21.351 ms 21.499 ms 21.681 ms]
throughput: [3.0227 Melem/s 3.0483 Melem/s 3.0695 Melem/s]
```

Mean moved from 21.735 ms to 21.499 ms, about 1.1 percent. That is below the
campaign keep threshold and inside shared-host Criterion noise.

`cargo flamegraph` was attempted, but `rch` classified it as non-compilation and
began a local build. The process was stopped and that run is not counted as proof.

## Verdict

Rejected by profile. Score: impact 0.5 x confidence 2 / effort 1 = 1.0.

The code lever was reverted manually. This pass records only the bead, golden
checksum, benchmark result, and rejected-pass rationale.
