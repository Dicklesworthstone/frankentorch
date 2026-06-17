# frankentorch-2i1cq closeout: BatchNorm1d row-major stats rejected

Target:
- `batch_norm/grad_1d_8192x1024`
- Profile source: `artifacts/perf/frankentorch-next-reprofile-20260617c/current_top_train_reprofile_after_16m8a.log`
- Post-GroupNorm reprofile timing: `[664.48 ms 670.59 ms 677.50 ms]`

Lever tried:
- Add a `spatial == 1` f64 `batch_norm_stats_f64` path that streams contiguous rows and updates per-channel accumulators in the same `n = 0..batch` order as the old channel-major loop.
- The candidate source was removed because the benchmark did not clear the keep threshold.

Behavior proof during trial:
- Kernel stats bit proof passed against the old channel-major reference.
- BatchNorm1d API gradient finite-difference and golden-bit tests passed.
- Candidate stats golden digest: `0x79d80a7ceb3580eb`.
- Ordering/RNG/isomorphism notes: no RNG or tie behavior; per-channel mean/variance addition order was preserved exactly; fallback surface outside `spatial == 1` was untouched.

Benchmark result:
- Baseline: `[690.38 ms 703.09 ms 715.15 ms]`
- Candidate: `[698.82 ms 713.73 ms 731.45 ms]`
- Criterion change: `[-1.1958% +1.5132% +4.8312%]`, `p = 0.36`
- Decision: reject. Median regressed to `0.9851x`, no significant win.
- Score: `0.00`

Next route:
- Stop iterating BatchNorm1d backward/stat micro-layout variants.
- Move to a different profile-backed primitive family, such as RMSNorm/LayerNorm all-ones backward stat staging or a pooling/SDPA structural pass.

Artifact hashes:
- `pass1_local_baseline_batch_norm1d_grad.log`: `cb7b2fff0a6695c6a78fb3aeb7e442b44c5ff0efbd9d196776d90ef29a71bdc0`
- `pass2_kernel_proof_batch_norm_stats_spatial1.log`: `75e18e6d05d38ac17ce60e53c4cc82a4ff9984e7a2d34e9fd20f311f2c5078ad`
- `pass2_api_batch_norm1d_grad_tests.log`: `8953d3a39949998c0fe47a250880b10769f7c5d17e22aada001b7672ca4e129f`
- `pass3_local_rebench_batch_norm1d_row_major_stats.log`: `b0190c46931d4358be58bfe4d19be594a21ef7bec72c5b59a479e10717103690`
- Reprofile artifact: `329f7857e6ea729756b49b377e4d160b96c180f57f973faa6855f21f3399fb0e`
