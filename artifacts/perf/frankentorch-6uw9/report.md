# frankentorch-6uw9 - borrowed-input conv2d custom autograd

## Target

- Bead: `frankentorch-6uw9`
- Profile-backed hotspot: `cargo bench -p ft-api --bench ops_bench -- conv2d/grad_hw/64`
- Baseline worker: `ts1`
- Baseline command: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- conv2d/grad_hw/64 --warm-up-time 1 --measurement-time 5 --sample-size 10`

## Lever

One lever: add a f64-only custom-autograd API that lets backward closures read immutable input slices from the tape, then route only the f64 `functional_conv2d` custom-op gradient path through it.

The old conv2d path saved owned copies of the padded input and weight:

- `ctx.save_for_backward(pv.to_vec(), ...)`
- `ctx.save_for_backward(wv.to_vec(), ...)`

The new path keeps the same forward and backward kernels but re-borrows the input slices at backward entry:

- forward: unchanged `ft_kernel_cpu::conv2d_forward_f64`
- backward: unchanged `ft_kernel_cpu::conv2d_backward_f64`
- input order: unchanged `[padded, weight, bias?]`
- gradient return order: unchanged `[dpadded, dweight, dbias?]`

## Isomorphism Proof

- Floating point: no arithmetic order changes inside either conv2d kernel; the same `dout`, padded-input slice, weight slice, dimensions, stride, and bias flag reach `conv2d_backward_f64`.
- Ordering/tie-breaking: no sorting, tie, or set-order behavior is touched.
- RNG: no RNG state or random values are touched.
- Autograd topology: the custom function still records the same input node IDs and completes the same dependencies; only the storage ownership of saved f64 inputs changes.
- Mutation contract: external leaf tensors requiring grad still reject in-place mutation before backward through `validate_tensor_in_place_target`. The benchmark padding path saves an internal padded tensor that `functional_conv2d` does not expose to callers. The focused golden test verifies that attempted pre-backward weight mutation fails before gradients are computed.
- Fallbacks: non-f64 paths, composed fallback paths, no-bias/bias shape handling, and non-conv2d custom functions keep the legacy owned-save API.

Golden output artifact: `artifacts/perf/frankentorch-6uw9/golden_conv2d_borrowed_contract.txt`.

## Benchmarks

Criterion, same worker `ts1`, same command shape.

| Run | Median | Interval |
| --- | ---: | --- |
| baseline | 223.02 ms | [209.64 ms, 236.13 ms] |
| after 1 | 209.29 ms | [193.09 ms, 227.75 ms] |
| after 2 | 177.43 ms | [166.58 ms, 188.63 ms] |
| after 3 | 172.56 ms | [160.28 ms, 185.52 ms] |

Kept score uses the median of the after medians, not the fastest run:

- baseline median: `223.02 ms`
- after median-of-medians: `177.43 ms`
- speedup: `1.26x`
- score: `Impact 3.0 x Confidence 0.90 / Effort 1.0 = 2.70`

## Validation

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo check -p ft-autograd --all-targets` passed.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo check -p ft-api --all-targets` passed.
- Post-rebase `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo check -p ft-api --all-targets` passed.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-autograd custom_function_borrowed_inputs_backward_uses_tape_values` passed.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-api functional_conv2d_borrowed_grad_preserves_input_contract` passed.
- Post-rebase `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-api functional_conv2d_borrowed_grad_preserves_input_contract` passed.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-api functional_conv2d_grad_matches_finite_diff` passed.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo clippy -p ft-autograd --all-targets -- -D warnings` passed.
- `cargo clippy -p ft-api --all-targets -- -D warnings` through RCH fails existing `ft-api` lint backlog; no diagnostics are attributable to the touched borrowed-input wrapper, conv2d call site, or new test.
- `cargo fmt --check` local fails existing workspace formatting backlog; `git diff --check -- crates/ft-autograd/src/lib.rs crates/ft-api/src/lib.rs` passed.
- `ubs crates/ft-autograd/src/lib.rs crates/ft-api/src/lib.rs` timed out after 120 seconds while scanning the two large files.

## Verdict

PRODUCTIVE. Keep.

Next primitive after re-profile: widen borrowed-input custom autograd to another large-input, profile-backed custom op only where mutation contracts are proven, or pivot to a deeper conv2d backward kernel primitive if profiler shows compute still dominates.
