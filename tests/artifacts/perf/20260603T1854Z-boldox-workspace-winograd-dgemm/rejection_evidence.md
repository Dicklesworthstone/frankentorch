# frankentorch-mdho Workspace-Backed Winograd/Strassen dgemm Rejection Evidence

## Target

- Crate: `ft-kernel-cpu`
- Benchmark: `cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Baseline primitive: existing row-split `matrixmultiply::dgemm`
- Attempted lever: one-level workspace-backed Strassen schedule for large power-of-two square f64 GEMM, with fused quadrant add/sub packing and seven parallel half-size subproducts

## Baseline

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Worker: `vmi1149989`
- Time: `[10.886 ms 11.176 ms 11.428 ms]`

## Proof

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu workspace_strassen_matches_reference_with_tolerance -- --nocapture`
- Worker: `vmi1227854`
- Result: passed
- Isomorphism checked: dtype/shape/error/RNG behavior unchanged; non-target GEMM shapes stayed on the original row-split path; target square f64 output order stayed row-major; floating-point reassociation was bounded by tolerance against the existing GEMM reference.

## After

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Worker: `vmi1227854`
- Time: `[20.845 ms 21.943 ms 23.330 ms]`

## Decision

- Rejected. The proof passed, but the lever was about 1.96x slower by p50 than the fresh baseline.
- Source lever and temporary proof test were manually reverted; no regressed GEMM code is kept.
- Diagnosis: scratch reuse removed recursive allocation churn, but the one-level Strassen representation still pays too much for quadrant packing, extra buffers, and recomposition relative to the current row-split microkernel.
- Next primitive: pivot to a different representation/parallelization model, such as Morton/Z-order tiled safe-Rust GEMM with portable SIMD register tiles and thread-local packed panels, or re-profile the ready perf queue and attack a non-GEMM hotspot if GEMM no longer scores highest.
- Score: impact -2 x confidence 4 / effort 2 = -4.0.
