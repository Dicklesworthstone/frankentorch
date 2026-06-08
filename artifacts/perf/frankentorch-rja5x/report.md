# frankentorch-rja5x - exact KNN SIMD distance panel

## Target

- Bead: `frankentorch-rja5x`
- Crate: `ft-api`
- Hotspot: `knn_search/8192x512_k8`
- Profile-backed residual: exact KNN still evaluates 4,194,304 point/query distances after the prior SoA point panel and k=8 top-k storage keeps.

## Lever

One lever only: for finite coordinate buffers in the existing `k == 8` KNN route, compute four point distances at a time with `wide::f64x4`, then replay the four lane distances into the unchanged `consider_knn_candidate_k8` helper in ascending lane order.

The non-finite coordinate case keeps the scalar route so NaN/Inf edge behavior remains conservative. Generic `k != 8`, shape validation, `k == 0`, `k > n_points`, output shape construction, and final `sqrt` output stay unchanged.

## Benchmark

Command:

```bash
rch exec -- cargo bench -p ft-api --bench ops_bench -- knn_search/8192x512_k8 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Same worker: `vmi1153651`

| Run | Median | Interval |
| --- | ---: | --- |
| Baseline, clean `475a5782` worktree | 17.337 ms | [14.814 ms, 20.527 ms] |
| SIMD panel | 8.9280 ms | [8.4860 ms, 9.6186 ms] |

Speedup: `1.9410x`

Score: `Impact 1.9410 x Confidence 0.80 / Effort 0.70 = 2.22`

Verdict: `KEEP` (`Score >= 2.0`, same-worker medians and intervals favor the candidate).

The retained raw logs used for the gate are `baseline_head_475a5782_retry1.log` and
`after_simd_retry1.log`. Earlier zero-byte or incomplete admission-pressure attempts
are intentionally excluded from the score.

## Isomorphism Proof

- Ordering preserved: yes. The route still iterates `q_start`, `p_start`, `local_q`, then ascending point lanes. SIMD lane results are replayed as lane `0, 1, 2, 3`.
- Tie-breaking unchanged: yes. `consider_knn_candidate_k8` is unchanged and still uses strict `<`; equal distances keep earlier point order.
- Floating-point: finite route computes `dx * dx + dy * dy + dz * dz` per lane without `mul_add`, `powi`, or fast-math. Non-finite buffers fall back to the prior scalar loop.
- RNG: unchanged and absent from `knn_search`; benchmark inputs are deterministic.
- Shapes/errors: unchanged for 2D/3D compatibility, `k == 0`, and `k > n_points`.
- Golden outputs: `sha256sum -c artifacts/optimization/golden_checksums.txt` passed.

## Verification

- `cargo test -p ft-api knn_search -- --nocapture`: passed 3/3 on `vmi1167313`.
- `cargo check -p ft-api --all-targets`: passed remotely on `vmi1153651`.
- `sha256sum -c artifacts/optimization/golden_checksums.txt`: passed.
- `git diff --check`: passed.
- `cargo fmt -p ft-api -- --check`: blocked by broad pre-existing `ft-api` formatting drift outside this KNN hunk; the KNN-specific formatter complaint was manually fixed.
- `cargo clippy -p ft-api --all-targets -- -D warnings`: failed on broad pre-existing `ft-api` lint debt unrelated to this KNN hunk; no logged diagnostic references `wide::f64x4`, `knn_coords_are_finite`, or the distance-panel lane replay.
- `ubs crates/ft-api/src/lib.rs crates/ft-api/Cargo.toml Cargo.lock`: timed out after 300 seconds on the large `ft-api` file after entering the Rust scan; no UBS finding was emitted before timeout.

## Next Shifted Target

Reprofile after this keep. If KNN remains the best ready profile-backed lane, the next materially different primitive should move beyond per-distance SIMD replay toward a cache-oblivious query/point layout or exact spatial index with deterministic full-scan tie ledger and scalar fallback for strict bit mode.
