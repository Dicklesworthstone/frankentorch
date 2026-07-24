# frankentorch-kgs4.54 pass 1: packed f64 dgemm_bt B panels

## Target

- Bead: `frankentorch-kgs4.54`
- Hotspot: `ft-kernel-cpu` f64 transposed-B GEMM path used by linear `x @ W^T`.
- Profile artifact: `pass1_profile_gemm_bt_ab.log`
- Profile-backed rows:
  - `linear x[512,1024] @ W[1024,1024]^T`: 9.331 ms, 115 GFLOP/s
  - `linear x[1024,1024] @ W[1024,1024]^T`: 16.293 ms, 132 GFLOP/s
  - `linear x[2048,512] @ W[2048,512]^T`: 38.703 ms, 111 GFLOP/s

## Lever

Pack each f64 transposed-B output column panel once per `j` tile in
`dgemm_bt_2d_parallel`, then reuse the packed logical `[k,bj]` panel across all
parallel `m` tiles. This removes repeated strided B-row reads for the same
output-column panel while keeping matrixmultiply's K accumulation order intact.

## Benchmark

Criterion row: `matmul_bt_f64_1024x1024x1024`.

- Initial baseline, different worker `vmi1152480`: `[25.550, 28.685, 31.637] ms`
  (`pass1_baseline_criterion_matmul_bt_f64_1024.log`)
- Same-worker clean baseline, `vmi1153651`: `[39.708, 46.763, 56.988] ms`
  (`pass1_clean_baseline_criterion_matmul_bt_f64_1024_vmi1153651_attempt.log`)
- After, `vmi1153651`: `[24.569, 27.108, 30.702] ms`
  (`pass1_after_criterion_matmul_bt_f64_1024.log`)
- After confirmation, `vmi1153651`: `[24.633, 27.261, 31.524] ms`
  (`pass1_after_confirm_criterion_matmul_bt_f64_1024_vmi1153651.log`)

Same-worker median speedup: `46.763 / 27.261 = 1.716x`.

Score: `Impact 1.716 * Confidence 0.90 / Effort 0.45 = 3.43`, keep.

## Isomorphism proof

- Ordering: K is not split; each `dgemm_mm` call still performs the same
  matrixmultiply accumulation order for each output tile.
- Tie-breaking: no comparisons or branch tie-breaks are introduced.
- Floating point: values loaded into `b_panel[kk * bj + jj]` equal old
  `b[(j0 + jj) * k + kk]`; the packed panel changes memory layout only, not
  arithmetic order or operands.
- RNG: no random state is introduced or consumed.
- Parallelism: output tiles remain disjoint by `(i0,j0)` ranges.

Proof commands:

- `cargo test -j 1 -p ft-kernel-cpu gemm_2d_parallel_is_bit_exact_vs_serial -- --nocapture`
  passed on `vmi1227854`; final rerun artifact:
  `pass1_isomorphism_gemm_2d_rerun.log`.
- `cargo run -j 1 -p ft-kernel-cpu --example gemm_golden --release` passed on
  `vmi1152480`; final rerun artifact: `pass1_golden_gemm_bt_rerun.log`.

Golden digest lines SHA256:

`18d10e542f61e0bc332a4e246dc5a950777301cddb00e66152c2c1f8f2399149`

Digest lines:

```text
512x512x512 fnv1a=b33fe4de6b0415b2 sum=-4.607643655265e2
512x512x512 bt_fnv1a=c564a4a701445a90 bt_sum=-5.983234880383e2
300x257x259 fnv1a=e12e548ed1f3227c sum=1.690899416496e2
300x257x259 bt_fnv1a=c9c73d9e883dc11b bt_sum=1.544677133058e2
```

## Quality gates

- `cargo check -j 1 -p ft-kernel-cpu --all-targets` passed on `vmi1152480`
  (`pass1_check_ft_kernel_cpu.log`).
- `cargo clippy -j 1 -p ft-kernel-cpu --all-targets -- -D warnings` passed on
  rerun on `vmi1152480` (`pass1_clippy_ft_kernel_cpu_rerun.log`).
- `rustfmt --check` passed for touched bench/golden files
  (`pass1_rustfmt_touched_files.log`).
- `git diff --check` passed (`pass1_git_diff_check.log`).
- UBS scan reported no critical findings.
