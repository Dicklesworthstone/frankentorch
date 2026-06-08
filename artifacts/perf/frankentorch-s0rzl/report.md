# frankentorch-s0rzl - KNN cache-oblivious/spatial follow-up

## Target

- Bead: `frankentorch-s0rzl`
- Crate: `ft-api`
- Hotspot: `knn_search/8192x512_k8`
- Current residual: exact KNN still evaluates 4,194,304 point/query pairs after the retained fixed k=8 top-k route, staged SoA point panel, and finite `f64x4` distance lanes.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1153651 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

RCH selected worker: `vmi1167313`

```text
knn_search/8192x512_k8  time:   [8.0757 ms 8.4273 ms 8.7614 ms]
                        thrpt:  [478.73 Melem/s 497.71 Melem/s 519.37 Melem/s]
```

This worker is the comparison gate for any candidate rebench in this bead.

## Prior KNN Families

- Rejected: scalar partial-distance threshold pruning (`frankentorch-swbh`).
- Rejected: query panel width tuning (`frankentorch-bbgu`).
- Rejected: point-major all-query top-k workspace (`frankentorch-jod6`).
- Kept: staged SoA point-coordinate panel (`frankentorch-iawv`).
- Kept: fixed k=8 top-k/register route.
- Kept: finite `wide::f64x4` point-distance lanes (`frankentorch-rja5x`).

## Candidate Gate

Next lever must be materially different from the rejected families. Candidate space is restricted to exact KNN structural primitives grounded in:

- Graveyard §7.11 nearest-neighbor retrieval: exact rerank dominated by SIMD distance kernels and SoA storage.
- Graveyard §8.2 vectorized execution: cache-sized batches and row-isomorphism proof.
- Graveyard §7.2 cache-oblivious layouts: recursive/locality-aware layout when cache behavior dominates.

No approximate ANN, no changed output ordering, no changed distance expression, and no changed error behavior may ship.
