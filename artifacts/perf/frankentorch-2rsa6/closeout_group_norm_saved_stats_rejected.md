# frankentorch-2rsa6 closeout

Decision: REJECT.

Profile-backed target:
- Source: `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`
- Row: `group_norm/grad_32x256x28x28` `[482.72 ms 493.98 ms 505.54 ms]`

Dedicated local baseline:
- Command: `RCH_REQUIRE_REMOTE=0 CARGO_TARGET_DIR=/data/tmp/frankentorch-next-reprofile-local-target cargo bench -j 1 -p ft-api --bench ops_bench -- 'group_norm/grad_32x256x28x28' --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot`
- Result: `[502.78 ms 509.77 ms 516.70 ms]`
- Artifact: `artifacts/perf/frankentorch-2rsa6/pass1_local_baseline_group_norm_grad.log`
- SHA-256: `a0695e47c8019687ab48e8ec947c7c1e2430930b5d07301f6f3d1ce5b10689a1`

Candidate lever:
- Added `group_norm_forward_with_stats_f64` and `group_norm_backward_with_stats_f64`, routing the f64 affine grad custom op through saved per-`(batch, group)` mean/rstd sidecars.
- Preserved output order, per-group scan order, mean/rstd floating-point bits, affine gradient serial accumulation order, dtype/error behavior, RNG absence, and f32/non-affine fallbacks.

Behavior proof:
- Kernel proof: saved-stats forward/backward matched recompute forward/backward bit-for-bit.
- Golden digest: `0x24b8c1646a209c37`.
- Artifact: `artifacts/perf/frankentorch-2rsa6/pass2_kernel_proof_group_norm_saved_stats.log`
- SHA-256: `d2155c33e6175dfebcbb7373161f386332310bddc0694b3babf2f241bc3b4e9c`
- API proof: `functional_group_norm` focused tests passed, 5/5.
- API artifact SHA-256: `e4605a952b138bcb3fb39697b2dcafb7c7d17e912f0c2354dc3f3c82acbd94d5`

Rebench:
- Original target dir was locked for more than 60 seconds, so the candidate used an isolated local target dir: `/data/tmp/frankentorch-2rsa6-rebench-target`.
- Candidate result: `[482.41 ms 494.19 ms 507.13 ms]`
- Baseline median to candidate median: `509.77 ms -> 494.19 ms`, `1.03x`.
- Artifact: `artifacts/perf/frankentorch-2rsa6/pass3_local_rebench_group_norm_saved_stats.log`
- SHA-256: `aaf594c4618abb643d6b91cd5273547957c8ce6779356c4e8dc91b2398308db3`

Score:
- `1.25 = Impact 1.03 * Confidence 0.85 / Effort 0.70`
- Below the `2.0` keep threshold.

Closeout:
- Source hunk removed.
- Do not repeat saved-stat-only GroupNorm variants.
- Next route: attack a larger norm primitive, such as fused norm backward with affine-gradient accumulation into upstream consumers, or move to the next profile row (`batch_norm/grad_train_32x256x28x28`, `layer_norm/grad_2048x1024`, or `rms_norm/grad_2048x1024`) with an algorithmic lever rather than another local rescan trim.
