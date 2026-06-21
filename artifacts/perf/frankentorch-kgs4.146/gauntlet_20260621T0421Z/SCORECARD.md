# frankentorch-kgs4.146 Scorecard

## Target

- Lane: `avg_pool1d` f64 scalar-sum backward, `[N,C,L]=[8,64,8192]`,
  `kernel=2`, `stride=2`.
- PyTorch gap before this probe: final FrankenTorch fused scalar row still
  loses to local PyTorch by about `2.13x`.
- Radical lever source: alien-graveyard/alien-artifact affine-loop
  specialization. In exact non-overlap coverage, every input receives the same
  scalar gradient, so the scatter can be algebraically collapsed.

## Candidate

Temporary candidate in `avg_pool1d_backward_scalar_f64`:

```rust
if stride == kernel && output_len.saturating_mul(stride) == len {
    let g = 0.0f64 + upstream / kernel as f64;
    return vec![g; batch * ch * len];
}
```

The `0.0 +` preserves the materialized accumulation behavior for signed zero
and NaN cases better than a raw assignment. The candidate was reverted after the
same-worker regression below.

## Evidence

| Check | Location | Result |
| --- | --- | --- |
| Bit behavior | RCH `ovh-a` | `cargo test -p ft-kernel-cpu avg_pool1d_sum_scalar_backward_matches_materialized_bits --lib -- --nocapture` passed |
| Candidate routing | RCH `hz2` | standard `101.57 ms`, fused scalar `65.018 ms`; remote PyTorch missing `torch` |
| Disabled baseline | RCH `ovh-a` | standard `51.218 ms`, fused scalar `27.461 ms`; remote PyTorch missing `torch` |
| Re-enabled candidate | RCH `ovh-a` | standard `52.488 ms`, fused scalar `48.523 ms`; fused scalar regressed `+77.804%`, `p = 0.00` |
| PyTorch comparator | local Python | median `12.898618498 ms/iter` over five 40-iteration totals |
| Post-revert check | RCH `ovh-a` | `cargo check -p ft-kernel-cpu --lib` passed |
| Conformance | RCH `vmi1227854` | `cargo test -p ft-conformance --profile release` passed; full crate, binaries, integration, smoke, and doctests green |

## Ratios

- Final reverted fused FT vs local PyTorch:
  `27.461 / 12.898618498 = 2.13x` slower.
- Candidate fused FT vs local PyTorch:
  `48.523 / 12.898618498 = 3.76x` slower.
- Candidate vs same-worker disabled fused baseline:
  `48.523 / 27.461 = 1.77x` slower.
- Win/loss/neutral vs PyTorch for this attempted lever: `0W / 1L / 0N`.

## Verdict

Rejected and reverted. The scalar-fill idea collapses arithmetic but sacrifices
the existing `par_chunks_mut` per-plane parallelism, so the large write becomes
serial and loses badly. Do not retry serial constant-fill shortcuts for this
lane.

Retry only with a design that keeps fill bandwidth parallel or removes the
allocation path entirely. The next serious avg_pool1d attempt should compare a
parallel constant-fill primitive or allocator/cache reuse against the current
parallel scatter on the same RCH worker, then report the fair PyTorch ratio.
