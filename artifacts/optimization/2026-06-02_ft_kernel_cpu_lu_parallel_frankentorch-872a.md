# ft-kernel-cpu LU Trailing-Update Parallel Pass

- Bead: `frankentorch-872a`
- Parent umbrella: `frankentorch-kgs4`
- Skills: `/extreme-software-optimization`, `/alien-graveyard`, `/alien-artifact-coding`
- Crate: `ft-kernel-cpu`
- Target benchmarks:
  - `det_lu_f64_768x768`
  - `det_lu_f64_1536x1536`

## Profile Target

`det_contiguous_f64` routes through `lu_factor_contiguous_f64`. For large
matrices, the O(n^3) trailing-submatrix update dominates runtime and is the
safe-Rust LAPACK-class gap this pass targets.

Baseline:

```text
worker: vmi1156319
command: rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench linalg_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 10
det_lu_f64_768x768: [200.83 ms 205.76 ms 210.95 ms]
det_lu_f64_1536x1536: [2.2017 s 2.2588 s 2.3133 s]
```

After:

```text
worker: vmi1156319
command: rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 10
det_lu_f64_768x768: [68.532 ms 71.105 ms 73.273 ms]
det_lu_f64_1536x1536: [1.4990 s 1.5534 s 1.6117 s]
```

Delta:

- 768x768 determinant/LU p50: `205.76 ms -> 71.105 ms`, about 2.89x faster.
- 1536x1536 determinant/LU p50: `2.2588 s -> 1.5534 s`, about 1.45x faster.
- Score: Impact 4 x Confidence 4 / Effort 2 = 8.0.

## Alien Recommendation Card

Change: parallelize the LU trailing-submatrix row updates after each pivot,
splitting the rows below the pivot into independent Rayon morsels while keeping
the pivot search, row swaps, and pivot order serial.

Mapped graveyard sections:

- High-level summary vectorized/morsel execution: split independent data-plane
  work into parallel batches where the chunk boundary matches ownership.
- Alien graveyard benchmark appendix: record the comparator and keep the change
  only if constants win in measured runs.
- Numerical linear algebra artifact family: preserve error and pivot semantics
  while introducing blocked/parallel kernels one lever at a time.

Expected value: Impact 4 * Confidence 4 * Reuse 4 / Effort 2 /
AdoptionFriction 1 = 32.0.

Fallback: keep the serial row-update path below `LU_PAR_MIN_ROWS` and whenever
the Rayon pool has one thread. Revert this trailing-update lever if the bit-exact
LU proof, golden checksum, or Criterion score fails.

## Alien Artifact Proof

Selected family: certified independent-row parallelization for LU.

Proof obligations:

- Ordering: pivot search, pivot row swaps, and the outer `k` loop remain serial.
- Tie-breaking: pivot tie behavior is unchanged because the max-row scan is not
  parallelized.
- Floating point: for a fixed pivot row, each target row performs the same
  multiplier computation and the same in-row `j` update order as the serial
  loop. Parallelism occurs only between independent target rows.
- RNG: LU uses no RNG.
- Errors: shape/layout/storage validation remains before factorization.
- Golden output: selected pivots and LU bit patterns for a 128x128 identity
  matrix are pinned by sha256
  `49832339f1bdc6ad678237d54763fb158fb99ac1942d600fe591198fc183158a`.

## Gates

- `rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-kernel-cpu --bench linalg_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 10` passed.
- `rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 10` passed.
- `rch exec -- cargo test -p ft-kernel-cpu lu_factor_ -- --nocapture` passed.
- `rch exec -- cargo check -p ft-kernel-cpu --all-targets` passed.
- `rch exec -- cargo clippy -p ft-kernel-cpu --all-targets --no-deps -- -D warnings` passed.
- `rch exec -- cargo fmt -p ft-kernel-cpu --check` passed.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed.
- `git diff --check` passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs crates/ft-kernel-cpu/Cargo.toml crates/ft-kernel-cpu/benches/linalg_bench.rs artifacts/optimization/2026-06-02_ft_kernel_cpu_lu_parallel_frankentorch-872a.md artifacts/optimization/golden_outputs/ft_kernel_cpu_lu_parallel_frankentorch-872a.txt artifacts/optimization/golden_checksums.txt` exited 0 with no critical findings.
