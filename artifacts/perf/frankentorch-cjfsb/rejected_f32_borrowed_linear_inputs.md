# frankentorch-cjfsb: rejected f32 Linear borrowed inputs

Bead: `frankentorch-cjfsb`

Attempted lever: in the no-grad f32 `FrankenTorchSession::functional_linear`
branch, borrow contiguous input, weight, and bias slices before
`ft_kernel_cpu::linear_tensor_f32` instead of cloning them with
`tensor_values_f32`.

Decision: reject. No source change is retained.

## Baseline

Worker: `vmi1227854`

Command:

```text
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
rch exec -- cargo bench -j 1 -p ft-api --bench ops_bench linear_forward/f32_hidden -- \
  --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Criterion medians:

| Case | Baseline |
| --- | ---: |
| `linear_forward/f32_hidden/1024` | 455.42 us |
| `linear_forward/f32_hidden/2048` | 660.98 us |

## Candidate validation

With the temporary borrowed-input edit applied:

```text
RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
rch exec -- cargo check -j 1 -p ft-api --lib --benches

RCH_WORKERS=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
rch exec -- cargo test -j 1 -p ft-api \
  functional_linear_f32_fused_matches_transpose_path_bit_exact -- --nocapture
```

Both passed. The focused test is the golden-output check: it compares the fused
f32 Linear path to explicit `weight.transpose(0, 1)` plus `tensor_addmm` by
`f32::to_bits()` for every output element.

## After measurements

The after measurements did not land on the baseline worker. They are retained
as routing evidence only, not keep/reject proof:

| Log | Worker | 1024 median | 2048 median |
| --- | --- | ---: | ---: |
| `after_ft_api_linear_f32_forward.log` | `vmi1152480` | 1.0544 ms | 1.2388 ms |
| `after_ft_api_linear_f32_forward_vmi122_attempt.log` | `vmi1152480` | 493.85 us | 1.5695 ms |

`RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854` still selected
`vmi1152480`, so no same-worker after exists.

## Transcript hashes

```text
93a970ca4e333f76ad985ec79613e8b6e3ca376e30b78dac5aa0bb7642ada5c7  baseline_ft_api_linear_f32_forward.log
9cd93242ab402eda3d807e03f2f04010344bcf044d342053b52181857bfff878  check_ft_api_lib_benches.log
0bf253fdc621168eece92d47123cf8db5e5042c3c3a34cd286bbf6394ca7bc23  test_f32_linear_bit_exact.log
7534f582e144c8ac21940060e492029948c74584e427f1a71c2e3a422cf0eb15  after_ft_api_linear_f32_forward.log
a0f7da22688647718a0c1e466266b67c3f0bbf2f0a6b533c513599e06c8a49cf  after_ft_api_linear_f32_forward_vmi122_attempt.log
```

## Isomorphism

The attempted source edit changed only input ownership. Arithmetic would still
be `linear_tensor_f32 -> gemm::sgemm_bt`, with identical output order, bias add
order, tie behavior, and RNG behavior. Since the measurements did not support a
keep, the source edit was removed.

## Next route

Do not continue this clone-removal micro-family. The next f32 Linear pass should
attack a deeper primitive: `sgemm_bt` itself, using a cache-blocked / packed-panel
safe-Rust SGEMM path or a same-process A/B harness that compares the current
safe SGEMM path against the new primitive under one RCH job.
