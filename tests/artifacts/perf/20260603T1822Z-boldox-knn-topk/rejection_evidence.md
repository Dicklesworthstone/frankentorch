# frankentorch-qxyu Rejection Evidence

## Target

`ft-api` KNN search benchmark:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 10
```

## Baselines

- Initial clean baseline on `vmi1293453`: `[23.508 ms 24.486 ms 25.496 ms]`.
- Additional clean detached-worktree baseline on `vmi1156319`: `[42.800 ms 44.208 ms 45.702 ms]`.

The spread confirms high worker variance for this target; the initial fast-worker baseline is the relevant comparison for the after run.

## Lever Tried

Replace the three KNN-local squared-distance calls:

```rust
(px - qx).powi(2) + (py - qy).powi(2) + (pz - qz).powi(2)
```

with explicit delta products:

```rust
dx * dx + dy * dy + dz * dz
```

## Behavior Proof

- Ordering unchanged: point scan, query scan, stable top-k insertion, and output order were unchanged.
- Tie/NaN comparator behavior unchanged: insertion still used the existing strict `partial_cmp == Some(Less)` rule.
- Floating-point intent unchanged: only each square primitive changed; the final addition order and `sqrt` output were unchanged.
- RNG unchanged: no RNG path exists in `knn_search`.
- Focused remote proof: `cargo test -p ft-api knn_search_streaming_topk_matches_full_sort_reference_bit_exact -- --nocapture` passed on `vmi1227854`.

## After

- After run on `vmi1227854`: `[24.311 ms 25.204 ms 26.003 ms]`.

This does not beat the fast-worker clean baseline (`24.486 ms` p50) and is well below the Score>=2.0 keep threshold. The source lever was reverted.

## Next Target

The prior full-sort to streaming top-k replacement already exists in `8285df15e`. The next KNN attempt should be a deeper structural kernel, not another scalar square micro-tune: e.g. batch multiple queries per point block to reuse point loads, or use a cache-blocked query x point tile that maintains per-query stable top-k buffers while reducing memory traffic.
