# frankentorch-kgs4.107 closeout: direct strided f64 softmax lanes

Status: rejected, source hunk removed.

Target: `ft-kernel-cpu` general strided f64 softmax/log_softmax, selected from `artifacts/perf/frankentorch-strided-softmax-20260615/`.

Current local baseline after the `ts1` remote override:

- Command: `env -u RCH_REQUIRE_REMOTE CARGO_TARGET_DIR=/data/tmp/frankentorch-a7xya-local-target cargo bench -j 1 -p ft-kernel-cpu --bench softmax_bench -- strided_4096x32x8_dim1 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- `softmax_f64_strided_4096x32x8_dim1`: `[2.0127 ms 2.0627 ms 2.1211 ms]`
- `log_softmax_f64_strided_4096x32x8_dim1`: `[2.0130 ms 2.0489 ms 2.0815 ms]`
- Artifact: `artifacts/perf/frankentorch-kgs4.107/pass1_local_baseline_strided_softmax.log`
- SHA-256: `63dff2e41c0ae00189f77c0bb0843efc07709a4ffaabaa29a9a079c25379b5fd`

One lever tested:

- Replace f64 general-strided gather-to-scratch + scatter with direct strided output writes.
- Add strided pairwise helpers that preserve the same `len <= 128` sequential leaves and `mid = len / 2` recursion as the contiguous `pairwise_sum_f64` tree.
- Leave f32 paths, last-dim fast paths, public APIs, shape/error behavior, and RNG/tie behavior unchanged.

Behavior proof:

- Command: `env -u RCH_REQUIRE_REMOTE CARGO_TARGET_DIR=/data/tmp/frankentorch-a7xya-local-target cargo test -j 1 -p ft-kernel-cpu softmax_family_parallel --lib -- --nocapture`
- Result: 2 tests passed.
- Proof coverage: bit-exact scratch-reference proof across fast and strided shapes, plus existing golden fixture.
- Artifact: `artifacts/perf/frankentorch-kgs4.107/pass2_proof_softmax_family.log`
- SHA-256: `a7ce683ad5c1a6ac3686a48f0ae3dffd09a59f891e33b160be90930fec1a3287`

Rebench:

- Command: `env -u RCH_REQUIRE_REMOTE CARGO_TARGET_DIR=/data/tmp/frankentorch-a7xya-local-target cargo bench -j 1 -p ft-kernel-cpu --bench softmax_bench -- strided_4096x32x8_dim1 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- `softmax_f64_strided_4096x32x8_dim1`: `[1.9734 ms 2.0357 ms 2.0966 ms]`, change `[-5.6621% -2.1392% +1.4715%]`, `p = 0.28`, no significant change.
- `log_softmax_f64_strided_4096x32x8_dim1`: `[1.9492 ms 1.9898 ms 2.0472 ms]`, change `[-4.5011% -1.8981% +0.9478%]`, `p = 0.23`, no significant change.
- Artifact: `artifacts/perf/frankentorch-kgs4.107/pass3_local_rebench_strided_softmax.log`
- SHA-256: `72fe6ed2770ff0c5d89a2a692b2da8fcaf5daf85a9b5a25f7292d944663e2077`

Score:

- Best median ratio: `1.030` on log_softmax (`2.0489 ms / 1.9898 ms`).
- Confidence: `0.35`, because both Criterion rows reported no significant change.
- Effort: `0.50`.
- Score: `0.72`, below the `2.0` keep threshold.

Decision:

- Reject.
- Remove the source hunk.
- Route away from scratch-elision variants on this already-optimized row.
