# frankentorch-rd1s pass 10: BLAS-3 lower rank-2k update primitive

## Target

- Bead: `frankentorch-rd1s`
- Primitive: blocked `dsytrd` trailing update
- Graveyard match: communication-avoiding / BLAS-3 dense-kernel conversion.
- No-gaps contract: build the safe-Rust LAPACK-class primitive rather than
  retrying EISPACK row fan-out.

## Change

Added `symmetric_rank2k_lower_update_f64(n, k, v, w, a)`:

```text
A := A - (V @ W^T + W @ V^T)
```

`V` and `W` are row-major `n x k` panels and `A` is row-major `n x n`. The
function updates only the lower triangle, matching the current symmetric
eigensolver storage contract. This pass does not route strict public `eigh`
through the reassociated BLAS-3 path yet; pass 11 is the blocked-panel
integration/fallback gate.

## Benchmark

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- 'sym_rank2k_lower_(scalar|gemm)_f64_256x32' --warm-up-time 1 --measurement-time 5 --sample-size 20
```

RCH selected worker `vmi1227854`.

Results:

- scalar lower update: `[792.39 us 823.98 us 864.56 us]`
- BLAS-3/GEMM lower update: `[257.67 us 260.74 us 264.13 us]`
- Median speedup: `3.16x`
- Score: `Impact 3.5 x Confidence 4.0 / Effort 1.5 = 9.3`, keep.

## Proof

Focused correctness:

- `cargo test -p ft-kernel-cpu symmetric_rank2k_lower_update_matches_scalar_reference -- --nocapture`

Quality gates:

- `cargo check -p ft-kernel-cpu --all-targets`
- `cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`
- `cargo fmt -p ft-kernel-cpu --check`

Golden public eigensolver payloads:

- before: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- remote after: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- local after: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`

Isomorphism:

- Ordering preserved: existing `eigh` route and `total_cmp` sort unchanged.
- Tie-breaking unchanged: existing eigen-pair ordering route unchanged.
- Floating-point: new helper intentionally reassociates only inside the new
  blocked-update primitive; strict public eigensolver golden remains unchanged.
- RNG: N/A.

## Next Gate

Pass 11 must integrate a blocked `dsytrd` panel behind an explicit strict fallback.
If the panel route changes golden output or eigenvector sign/order without an
accepted ledger, keep the current scalar EISPACK route as the public strict path.
