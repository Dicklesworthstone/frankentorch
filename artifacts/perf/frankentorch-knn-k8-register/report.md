# frankentorch-78tey KNN k=8 register top-k path

## Target

`br ready --json` returned no ready perf beads after `frankentorch-r6cxt`, and
the visible in-progress perf lanes were already claimed by other agents. This
pass used the existing profile-backed KNN lane:

- `frankentorch-tgst` kept fixed-size top-k maintenance for
  `knn_search/8192x512_k8`.
- `frankentorch-swbh` rejected scalar partial-distance threshold pruning.
- `frankentorch-bbgu` rejected query-panel widening.
- `frankentorch-jod6` rejected all-query point-major workspace.

Residual primitive: a k=8 register-friendly top-k update path over the existing
cache-tiled query/point panels. This is the benchmark-dominant API shape and
preserves the exact exhaustive point scan and strict top-k contract.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

RCH selected worker `vmi1149989`.

```text
knn_search/8192x512_k8  time: [5.5382 ms 5.7932 ms 6.1373 ms]
```

Baseline log: `baseline_knn_search_ts1.log`.

## One Lever

Specialize only `k == 8` in `FrankenTorchSession::knn_search`:

- Use fixed `[[usize; 8]; KNN_QUERY_TILE]` and
  `[[f64; 8]; KNN_QUERY_TILE]` buffers.
- Use a fixed-array `consider_knn_candidate_k8` helper.
- Keep the existing generic slice/length path for every other `k`.

No benchmark, API, dtype, shape, autograd, or error-surface changes were made.

## Isomorphism Proof

- Ordering: batch order, query-tile order, local query order, point-panel order,
  and per-query point index scan remain unchanged.
- Tie-breaking: candidates still insert only on
  `partial_cmp(...) == Some(Ordering::Less)`. Equal distances keep earlier point
  order, and equal-to-worst candidates are rejected once the top-k buffer is
  full.
- Floating point: every candidate still evaluates
  `dx * dx + dy * dy + dz * dz`, and retained outputs still apply `sqrt()` only
  after top-k selection. The fixed-array helper changes storage shape, not the
  distance expression.
- RNG: `knn_search` contains no RNG, and the benchmark fixture is deterministic.
- Golden output: `sha256sum -c artifacts/optimization/golden_checksums.txt
  --ignore-missing` passed for all locally present tracked outputs, including
  KNN goldens.

Focused proof:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api knn_search -- --nocapture
```

Result: 3/3 focused KNN tests passed on `vmi1156319`, including the bench-scale
full-sort bit-exact test.

Check:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-api --all-targets
```

Result: passed on `vmi1156319`.

## Rebench

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Same worker: `vmi1149989`.

```text
knn_search/8192x512_k8  time: [4.9067 ms 5.2413 ms 5.5962 ms]
```

Median speedup: `5.7932 / 5.2413 = 1.105296x`.

Score: `Impact 1.105296 x Confidence 0.97 / Effort 0.45 = 2.38`, above the
`>= 2.0` keep gate.

## Gates

- PASS: `git diff --check`.
- PASS: focused KNN tests.
- PASS: golden SHA-256 verification.
- PASS: `cargo check -p ft-api --all-targets`.
- BLOCKED: `cargo fmt -p ft-api --check` reports broad pre-existing formatting
  drift across `ft-api` source, benches, and examples outside this KNN hunk.
- BLOCKED: `cargo clippy -p ft-api --all-targets -- -D warnings` reports broad
  pre-existing `ft-api` clippy debt: 191 lib errors and 216 lib-test errors.
- INCONCLUSIVE: `timeout 300 ubs crates/ft-api/src/lib.rs` produced only the
  UBS scanner banner and no findings before the timeout/status wrapper failed on
  the large-file scan.

## Verdict

Keep. The source hunk is one isolated k=8 storage/update specialization for the
profile-backed KNN benchmark, with unchanged observable behavior and same-worker
Criterion improvement.

Next profile route: re-run `br ready --json`. If no ready perf bead appears and
KNN remains the best disjoint lane, the next deeper primitive is a genuine
distance-panel algorithm change: portable SIMD or a cache-oblivious point/query
layout that computes multiple candidate distances per loaded coordinate while
preserving strict point order and bitwise distance output.
