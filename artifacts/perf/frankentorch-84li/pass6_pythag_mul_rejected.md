# frankentorch-84li pass 6: `eigh_pythag` direct-square rejection

## Target

Profile-backed residual: full-vector symmetric eigensolver remains slower than the packed values-only path after `frankentorch-a9ry`.

Baseline same-worker Criterion (`RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- 'eigh_f64_256x256|eigvalsh_f64_256x256' --warm-up-time 1 --measurement-time 5 --sample-size 20`):

- `eigh_f64_256x256`: `[16.699 ms 16.746 ms 16.808 ms]`
- `eigvalsh_f64_256x256`: `[9.4280 ms 9.4513 ms 9.4802 ms]`

Profile-time evidence (`pass6_profile_time_eigh_256.txt`): Criterion profiled `eigh_f64_256x256` for 5 seconds on `ts2`; raw `perf stat` was refused by `rch` remote-required policy because it wraps the build in `bash`.

## Candidate

One lever: replace the two `.powi(2)` squares in `eigh_pythag` with `ratio * ratio`.

Rationale: `eigh_pythag` is used inside the tridiagonal QL / Hessenberg-style rotation streams; direct multiplication should remove generic integer-power overhead while preserving the nonnegative-ratio arithmetic path.

## Isomorphism Proof

- Ordering preserved: yes. The sort key remains `f64::total_cmp`, and this lever does not touch pair construction or permutation.
- Tie-breaking unchanged: yes. Stable `sort_by` behavior and original index association are untouched.
- Floating-point: rejected despite the fixed fixture matching bit-for-bit; source was removed, so the kept tree returns to the original `.powi(2)` operation.
- RNG seeds: N/A.
- Golden outputs: clean full-eigh fixture before and candidate both produced SHA-256 `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`.

## Candidate Benchmark

Same worker and same Criterion settings:

- `eigh_f64_256x256`: `[16.653 ms 16.702 ms 16.762 ms]`
- `eigvalsh_f64_256x256`: `[9.4754 ms 9.5080 ms 9.5528 ms]`

## Verdict

Rejected.

The full-vector median moved from `16.746 ms` to `16.702 ms` (about 1.003x), while `eigvalsh_f64_256x256` regressed from `9.4513 ms` to `9.5080 ms` (about 0.994x). Score is below the `Impact x Confidence / Effort >= 2.0` keep gate.

Next primitive: structural full-eigh memory movement around eigenvector permutation/layout first, then the deeper safe-Rust LAPACK-class plan: blocked symmetric tridiagonalization plus tridiagonal divide-and-conquer / secular merge with deterministic deflation and sign-orientation proof.
