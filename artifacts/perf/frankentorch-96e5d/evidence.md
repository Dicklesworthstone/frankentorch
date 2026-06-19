# frankentorch-96e5d — lazy Sum/Mean backward accumulation

BlackThrush · 2026-06-19 · ft-autograd · bit-exact core-path win

## Lever (ONE)

`TensorGradientTape::backward_with_options` Sum and Mean arms materialized a full
`vec![grad_scalar; input_numel]` (resp. `vec![grad_scalar*scale; numel]`) constant
contribution, then fed it to `accumulate_tensor_gradient` (`target[i] += contrib[i]`).
That buffer is pure alloc + fill + read traffic on the **universal `loss.backward()`
reduction** (loss is almost always `x.sum()` / `x.mean()`). Replaced both with the
existing, rao3v-proven lazy `accumulate_tensor_gradient_with(input, target, numel, |_| c)`
closure — no materialized Vec. Bit-identical arithmetic, same ascending index order,
same f64 op.

## Root-cause that motivated the lever (avg_pool1d gauntlet lane, kgs4.122)

Phase-timing probe (`crates/ft-api/examples/avgpool1d_phase_timing.rs`), avg_pool1d
`[8,64,8192]` f64 sum-loss train step, per-iter (hz2, RAYON=32, contended):

| phase | time |
|---|---:|
| session_new | 0.9 µs |
| tensor_var (clone+node) | ~7–10 ms |
| forward (apply_function) | ~21 ms |
| sum | ~0.6 ms |
| **backward** | **~70–134 ms (dominates, 75%)** |
| RAW avg_pool1d_forward_f64 kernel | 2.2–2.9 ms |
| RAW avg_pool1d_backward_f64 kernel | 2.9–8.0 ms |

**The pooling kernels are innocent (~3 ms). The 25x PyTorch gap lives in the GENERIC
autograd backward machinery.** Control tape `sum(x).backward()` on the same 4M leaf
(NO pooling op) = **35–53 ms** — pure generic machinery (grads alloc + Sum arm +
report/persistent). PyTorch does the whole step in ~7 ms. The wall is large fresh
buffer allocation / first-touch page-fault / serial bandwidth-bound copy, NOT the op.

## Measured win (same-process, same-worker A/B — avoids the rao3v worker-variance trap)

Pre-faulted reused target buffers, m = 4M f64, 64 reps, one process on one worker:

```
OLD fill+acc (vec![scalar;m] + target[i]+=contrib[i]): min 14941 µs / mean 15706 µs
NEW lazy acc (target[i]+=scalar, no contrib Vec)      : min  1088 µs / mean  1175 µs
ratio: 13.73x (min)  /  13.37x (mean)
```

The eliminated constant-contribution buffer was ~13.7x of the Sum-arm accumulation
cost (almost all of it allocation/fill/read of the throwaway 33 MB Vec). The lazy
path keeps only the unavoidable `target[i] += c` RMW.

> Note (honesty): a naive A/B that freshly allocates `target` each rep showed 0.73x —
> first-touch page faults of the fresh `target` swamp the arithmetic and invert the
> result. That is exactly the bandwidth/alloc-bound, order-dependent confound rao3v
> documented. The pre-faulted-buffer A/B isolates the real removed work.

## Verification gates (all GREEN)

| gate | command (rch, CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cc) | result |
|---|---|---|
| ft-autograd tests | `cargo test --release -p ft-autograd` | 476 passed, 0 failed |
| Conformance | `cargo test --release -p ft-conformance` | 199 + all sub-suites passed, 0 failed |
| Clippy | `cargo clippy --release -p ft-autograd -- -D warnings` | clean |
| fmt | `rustfmt --edition 2024 --check` (hunk) + `git diff --check` | clean |

## Why not parallelize the accumulate too

Parallelizing the pure `target[i] += c` RMW is memory-bandwidth-bound. The apparent
2.45x serial→rayon I first saw (4M `+=`: 21.7→8.85 ms) is the contended-single-thread
mirage rao3v/bandwidth-frontier notes warn about (single thread starved for bandwidth
under peer load; parallel grabs idle channels). NOT shipped — would falsely read as a
win. The genuine remaining ≥2x lever for these gauntlet lanes is a backward grad-buffer
scratch/caching allocator (gmuml-class: eliminate the per-backward fresh multi-MB
alloc+zero+page-fault storm), tracked separately.
