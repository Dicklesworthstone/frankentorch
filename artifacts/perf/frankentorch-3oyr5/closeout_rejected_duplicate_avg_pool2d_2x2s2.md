# frankentorch-3oyr5 closeout: rejected duplicate avg_pool2d 2x2s2 lane

Status: rejected/no code kept.

Finding:
- The profiled target was valid, but the intended direct 2x2 stride-2 avg-pool path and golden digest test were already present in `HEAD`.
- The only experimental delta in this pass was a temporary f32-path removal/f64-only candidate. That was restored before closeout.

Baseline, same-worker RCH `ovh-a` (`pass1_baseline_pool2d_ovh_a.log`):
- `max_pool2d/nograd`: `[1.5711 ms 1.6039 ms 1.6326 ms]`
- `max_pool2d/nograd_f32`: `[790.54 us 814.04 us 839.81 us]`
- `avg_pool2d/nograd`: `[1.7719 ms 1.8219 ms 1.8739 ms]`
- `avg_pool2d/nograd_f32`: `[1.0370 ms 1.0646 ms 1.0913 ms]`

Candidate attempts:
- Mixed candidate (`pass4_candidate_pool2d_ovh_a.log`) showed `avg_pool2d/nograd` median `1.4213 ms`, but `avg_pool2d/nograd_f32` widened/regressed to median `1.3524 ms`.
- f64-only repeat 1 (`pass6_candidate_f64_only_pool2d_ovh_a.log`) had contaminated controls: `max_pool2d/nograd_f32` median `4.6746 ms`; `avg_pool2d/nograd` median `3.5752 ms`.
- f64-only repeat 2 (`pass7_candidate_f64_only_pool2d_repeat_ovh_a.log`) repeated the contamination: `max_pool2d/nograd_f32` median `4.4531 ms`; `avg_pool2d/nograd` median `3.2376 ms`.

Isomorphism/golden evidence:
- Focused test: `rch exec -- cargo test -j 1 -p ft-kernel-cpu avg_pool2d_2x2s2_direct_matches_generic_bit_exact -- --nocapture`
- Result log: `pass5_test_avg_pool2d_f64_direct_digest_ovh_a.log`
- Golden digest artifact: `golden_avg_pool2d_2x2s2.txt`
- Golden sha256 check: `golden_avg_pool2d_2x2s2_sha256_check.log`
- SHA: `eb5df0a701ddafdcb18fee4d53af7fd92ad3954a17e74f3dc86804d3d8f3bb8e`

Decision:
- Score `< 2.0` because there is no new kept lever and the measured candidate is not confidence-grade.
- Close this bead as duplicate/rejected and reprofile for a different perf-backed primitive.
