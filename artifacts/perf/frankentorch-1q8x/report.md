# frankentorch-1q8x: f32 scan dispatch direct storage

## Target

- Bead: `frankentorch-1q8x`
- Surface: `tensor_cumprod` over f32 storage, `[8192, 1024]` along `dim=1`
- Lever: remove the f32 scan dispatch `Vec<f32> -> Vec<f64> -> Vec<f32>` round trip.

## Change

- `dispatch_tensor_scan_dim_contiguous_f32` now returns `TensorScanDimDispatchOutcomeF32`.
- The typed scan wrapper stores f32 scan results directly in `TensorStorage::F32`.
- The f64 scan path is unchanged.
- The f16/bf16 promoted path still promotes to f32 and returns f32 storage.

## Benchmark

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1293453 rch exec -- cargo bench -p ft-api --bench ops_bench -- cumprod/f32_nograd_8192x1024_dim1 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Same-worker Criterion on `vmi1293453`:

- Baseline: `[49.347 ms 50.254 ms 51.267 ms]`
- After: `[15.985 ms 16.573 ms 17.221 ms]`
- Median speedup: `3.03x`

Additional unpinned after run on `ts1`:

- After: `[23.553 ms 23.943 ms 24.443 ms]`

One earlier pinned after artifact, `after_f32_cumprod_direct_storage_vmi1293453.txt`, is retained but invalid for scoring because it produced no Criterion timing lines for the target filter.

## Proof

- Scan kernel loop/order unchanged.
- No floating-point reassociation, ordering, tie-breaking, or RNG behavior changed.
- Dispatch decision strings/keysets remain the same for f32 scan calls.
- `f32_cumsum_preserves_dtype` and `f32_cumprod_preserves_dtype` passed.
- `dispatch_typed_scan_preserves_f32_storage` passed.

## Gates

- `cargo check -p ft-dispatch -p ft-autograd -p ft-api --all-targets`: passed.
- `cargo clippy -p ft-dispatch -p ft-autograd --all-targets --no-deps -- -D warnings`: passed.
- Broad `cargo clippy -p ft-dispatch -p ft-autograd -p ft-api --all-targets -- -D warnings`: blocked by pre-existing `ft-kernel-cpu` dependency lints.
- Broad `cargo clippy ... --no-deps`: blocked by pre-existing ft-api lint debt.
- `cargo fmt --check`: blocked by pre-existing formatting drift across ft-api, ft-kernel-cpu, and ft-nn.
- `ubs` on touched files exited 1 with broad existing inventory findings; no new targeted finding in the scan dispatch change.

## Score

`Impact 3.03 x Confidence 0.95 / Effort 1.0 = 2.88`

Verdict: keep.
