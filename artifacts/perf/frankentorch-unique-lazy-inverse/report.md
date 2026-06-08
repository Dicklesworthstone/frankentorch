# frankentorch-idn91: tensor_unique lazy inverse bookkeeping rejected

## Target

- Bead: `frankentorch-idn91`
- Surface: `crates/ft-api/src/lib.rs`, `tensor_unique`
- Profile-backed benchmark: `unique/all_distinct_20000`

The benchmark calls `tensor_unique(t, sorted=false, return_inverse=false, return_counts=false)`.
The tested lever lazily allocated and filled inverse/count sidecars only when a caller requested
`return_inverse`, `return_counts`, or sorted sidecar remapping.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- unique/all_distinct_20000 --sample-size 20 --warm-up-time 1 --measurement-time 2
```

Concrete worker: `vmi1153651`

Criterion:

```text
unique/all_distinct_20000
time: [2.1932 ms 2.2539 ms 2.3242 ms]
```

## Candidate

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1153651 RCH_WORKERS=vmi1153651 rch exec -- cargo bench -p ft-api --bench ops_bench -- unique/all_distinct_20000 --sample-size 20 --warm-up-time 1 --measurement-time 2
```

Concrete worker: `vmi1153651`

Criterion:

```text
unique/all_distinct_20000
time: [2.5798 ms 3.0535 ms 3.4823 ms]
```

Same-worker median ratio: `2.2539 / 3.0535 = 0.738x` (regression).

## Behavior proof attempted

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api unique -- --nocapture`
  - Worker: `vmi1227854`
  - Result: pass, `14 passed; 0 failed`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-conformance fuzz_metamorphic_unique_contract -- --nocapture`
  - Result: blocked before running the target by pre-existing `ft-serialize` compile errors:
    `repair_equation` now returns a `Result`, and `ProofHash` no longer matches the local
    `u64`/lower-hex use sites.

## Isomorphism ledger

- Ordering: intended to preserve first-occurrence order for unsorted output and existing
  `total_cmp` sorted order.
- Tie-breaking: intended to preserve the stable `sort_by` behavior and repeated-NaN slots.
- Floating-point: no arithmetic or value conversion changes were intended; output `f64` bits
  were unchanged by construction.
- RNG: no RNG state or random generation path was touched. `HashMap` seeding remains unobserved
  because the map is not iterated.
- Error behavior: `requires_grad` rejection remained before value extraction/bookkeeping.

## Verdict

Rejected. Although proof-clean in focused `ft-api` tests, the same-worker Criterion result
regressed from `2.2539 ms` to `3.0535 ms`, so Score is `0.0`, below the `>= 2.0` keep gate.
The source hunk was removed and `crates/ft-api/src/lib.rs` is clean.

## Reroute

Do not continue adjacent `tensor_unique` bookkeeping micro-levers. The next target should be a
profile-backed structural primitive, preferably a deeper no-gaps safe-Rust kernel or layout
replacement that changes the dominant work rather than sidecar allocation.
