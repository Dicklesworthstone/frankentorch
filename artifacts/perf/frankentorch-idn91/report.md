# frankentorch-idn91 Rejection Report

## Target

- Bead: `frankentorch-idn91`
- Crate: `ft-api`
- Benchmark: `unique/all_distinct_20000`
- Candidate lever: lazily materialize `tensor_unique` inverse/count sidecars only when requested.

## Profile-Backed Baseline

Initial profile-backed baseline before editing:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- unique/all_distinct_20000 --sample-size 20 --warm-up-time 1 --measurement-time 2
worker: vmi1153651
time: [2.1932 ms 2.2539 ms 2.3242 ms]
```

The benchmark calls `tensor_unique(t, sorted=false, return_inverse=false, return_counts=false)`.
The observed code still allocated and filled `inverse_indices` for every input element.

## Same-Worker A/B

Because RCH worker selection shifted after the edit, the keep/reject gate used a detached
baseline worktree at `HEAD` and pinned both runs to `vmi1293453`.

```text
baseline worktree: /data/projects/.scratch/frankentorch-idn91-baseline-20260608T2135Z
baseline worker: vmi1293453
baseline time: [1.0209 ms 1.0356 ms 1.0528 ms]

candidate worker: vmi1293453
candidate time: [1.0185 ms 1.0461 ms 1.0780 ms]
```

Median ratio: `1.0356 / 1.0461 = 0.990x`. The candidate regressed slightly.

## Isomorphism Proof

- Ordering preserved: candidate preserved first-occurrence unsorted output order and sorted `total_cmp` output order.
- Tie-breaking unchanged: candidate preserved first occurrence for `+0.0`/`-0.0` and treated each NaN as distinct.
- Floating-point unchanged: candidate did not change output arithmetic; unique values, inverse indices, and counts stayed as integer-like sidecar construction in `f64`.
- RNG unchanged: `tensor_unique` uses no RNG.
- Golden behavior: focused unique tests passed on the candidate; final kept tree has no `crates/ft-api/src/lib.rs` source diff after rejection.

## Verdict

Rejected. The lazy sidecar materialization lever did not clear the `Score >= 2.0` keep gate and was removed.

Next primitive: a fundamentally different exact-f64 dedup engine, such as a compact open-addressed/Swiss-style safe-Rust table or sorted-cardinality split path, with a target ratio of at least `1.5x` on `unique/all_distinct_20000`.
