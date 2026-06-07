# frankentorch-jod6 KNN point-major workspace rejection

## Baseline/profile

Context: `br ready --json` returned no ready work after `frankentorch-bbgu`; open perf lanes `frankentorch-rd1s` and `frankentorch-kgs4` were already claimed by other agents. This follow-up stayed on the profile-backed KNN lane from `frankentorch-tgst`, `frankentorch-swbh`, and `frankentorch-bbgu`.

Rejected predecessor: `frankentorch-bbgu` proved that widening the existing query panel from 16 to 32 was the wrong family (`8.5562 ms -> 13.449 ms` on same-worker `ts1`, Score `1.01`). The next deeper primitive tested here was a point-major exact scan with a persistent top-k workspace for all queries.

Baseline command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

```text
knn_search/8192x512_k8  time: [7.9053 ms 8.1467 ms 8.5783 ms]
```

## Candidate

One lever only: replace per-16-query panels with one batch-wide top-k workspace:

- allocate `best_indices`, `best_distances`, and `best_lens` for all queries in the batch;
- scan each point once;
- update every query's strict top-k state for that point;
- emit the same sorted per-query top-k prefix after the point scan.

This preserved each query's point order (`pi` ascending), strict-less tie behavior, the exact `dx * dx + dy * dy + dz * dz` distance expression, and the same `sqrt` output path.

Pre-score: Impact `3.0` x Confidence `0.70` / Effort `1.0` = `2.10`.

## Proof

Focused proof:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-api knn_search -- --nocapture
```

Result on `ts1`: 3 KNN tests passed, including `knn_search_bench_scale_matches_full_sort_reference_bit_exact`.

Golden SHA-256:

```bash
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Result: all locally present golden outputs passed, including the KNN fixtures.

Isomorphism:

- Ordering: for each query, candidate distances are still considered by point index in ascending order; only cross-query interleaving changed.
- Tie-breaking: unchanged strict `<` in `consider_knn_candidate`; equal distances keep earlier point order.
- Floating-point: distance expression and `sqrt` output path unchanged; bench-scale full-sort digest passed bit-for-bit.
- RNG: no RNG path in `knn_search`.
- Autograd: KNN remains a value-only output path with `requires_grad=false`.

## Rebench

Same-worker after command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

```text
knn_search/8192x512_k8  time: [9.0127 ms 9.6224 ms 10.557 ms]
```

Median ratio: `8.1467 / 9.6224 = 0.847x`.

Score: Impact `0.85` x Confidence `0.95` / Effort `1.0` = `0.80`, below the `2.0` gate.

Verdict: rejected. The point-major workspace source hunk was removed; no source code was retained.

Next primitive: do not repeat query-panel width or all-query workspace tuning. Route to a staged coordinate-panel / SoA primitive that improves the distance scan's memory layout without increasing the hot top-k workspace footprint.
