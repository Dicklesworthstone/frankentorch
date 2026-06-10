# Pass 11 Rejected Routes And Next Profile

## State

- Worktree: `/data/projects/frankentorch-5oqum-boldfalcon`
- HEAD: `a708c9e3` (`perf(ft-kernel-cpu): cap staged eig band panel width`)
- Bead: `frankentorch-5oqum`, status `in_progress`, assignee `BoldFalcon`
- Behavior kept: none. All pass11 source hunks were removed after proof or benchmark gates failed.

## Baseline

Pass10 same-worker `vmi1227854` baseline:

| Row | Median |
| --- | ---: |
| `eigvalsh_f64_256x256` | 6.4720 ms |
| `eigvalsh_two_stage_f64_256x256_b32` | 11.491 ms |
| values-only stage1 harness `n256 b32` | 3542.65 us |

The profile-backed target remains the roughly 7.95 ms post-stage1 residual in
the staged path at `n256 b32`.

## Rejected Candidates

### Row-band storage for `banded_to_tridiagonal_f64`

- Proof: `pass11_rowband_bit_exact_test_remote_retry.log`
- Benchmark: `pass11_rowband_banded_to_tridiag_remote.log`
- Result: bit-exact, but `banded_to_tridiag_f64_256x256_b32` was `[21.170 ms 21.554 ms 22.102 ms]` versus pass10 `21.868 ms` median.
- Decision: rejected. The speedup was about `1.01x`, below the Score gate and still slower than the staged path.

### Sparse finite-band packed Householder reducer

- Proof: `pass11_sparse_active_test_remote_retry.log`
- Result: both focused fixtures rejected the sparse route before benchmarking:
  - `stage1-band n=32 b=4: sparse finite band reducer rejected fixture`
  - `clustered-finite-band: sparse finite band reducer rejected fixture`
- Decision: rejected. The source hunk was removed; no performance claim made.

### Row-contiguous symmetric matvec inside `eigh_tred2_values_only`

- Proof: `pass11_row_sweep_bit_exact_test_remote_retry2.log`
- Result: bit-exact against the old packed-column walk.
- Same-worker benchmark: `pass11_row_sweep_rebench_remote.log`
  - `eigvalsh_f64_256x256`: `[8.9015 ms 9.0879 ms 9.3024 ms]`
  - `eigvalsh_two_stage_f64_256x256_b32`: `[12.650 ms 12.811 ms 12.997 ms]`
- Decision: rejected. It regressed the staged row versus the pass10 median `11.491 ms`, despite preserving behavior exactly.

## Isomorphism Ledger

- Ordering/ties: final sort policy stayed `f64::total_cmp` in all candidates.
- Floating point: row-sweep was bit-exact by construction and test; sparse finite-band had tolerance proof only but failed fixture acceptance; row-band Givens was bit-exact against its full-window oracle.
- RNG: none.
- Public golden: public dispatch was unchanged; no source hunk survived pass11.

## Next Target

Do not continue storage-only or scalar-loop reshuffling. The next pass should
split-profile the staged residual at symbol or private-stage level and then
attack one of these deeper primitives:

1. values-only tridiagonal secular/divide-and-conquer or bisection-family solver
   if QL dominates the residual;
2. true compact-WY/dsytrd panel update if `eigh_tred2_values_only` dominates;
3. a real DSBTRD-style band reduction only if profiling shows it can beat the
   current dense packed reducer, not just the old standalone Givens helper.

Acceptance remains same-worker `vmi1227854`, Score `>= 2.0`, focused eigensolver
tests, unchanged public golden SHA until public dispatch is intentionally moved,
and no public `eigvalsh/eigh` wiring unless staged beats live in the same run.
