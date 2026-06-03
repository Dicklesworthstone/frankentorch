# frankentorch-5ni8 Packed-Panel dgemm Rejection Evidence

## Target

- Crate: `ft-kernel-cpu`
- Benchmark: `cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Baseline primitive: existing row-split `matrixmultiply::dgemm`
- Attempted lever: safe-Rust 4-column packed B panels reused across 4-row `wide::f64x4` register tiles

## Baseline

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Worker: `vmi1156319`
- Time: `[24.045 ms 25.633 ms 28.166 ms]`

## Proof

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu gemm_row_split_matches_single_bit_exact -- --nocapture`
- Worker: `vmi1227854`
- Result: passed
- Isomorphism checked: row splitting preserved deterministic per-element accumulation for the attempted packed-panel dgemm path.

## After

- Command: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench gemm_bench -- matmul_f64_1024x1024x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10`
- Worker: `vmi1153651`
- Time: `[77.658 ms 90.043 ms 103.03 ms]`

## Decision

- Rejected. The packed-B-panel/4x4 safe-Rust kernel is still slower than the existing baseline, with a roughly 3.5x p50 regression.
- Source lever was manually reverted; no regressed GEMM code is kept.
- Diagnosis: packing four columns of B is too narrow and does not overcome the throughput gap. The next primitive should change algorithmic structure, not tune this microkernel again.
- Next primitive: recursive Strassen/Winograd-style safe-Rust dgemm pilot for power-of-two large square matrices, gated by deterministic tolerance/golden fixtures and a cutoff fallback to the existing exact row-split GEMM.
- Score: impact -2 x confidence 4 / effort 1 = -8.0.
