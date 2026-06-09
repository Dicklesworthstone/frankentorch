# frankentorch-34hx9 pass 4: blocked values-only reducer rejected

## Candidate

The candidate implemented a values-only blocked Householder reducer for
`eigvalsh`. It delayed the fully leading top-left panel update and applied it
with the existing GEMM-backed lower rank-2k primitive, while updating same-panel
reflector rows immediately.

## Correctness probe

Focused scalar-versus-blocked proof passed before benchmarking:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a rch exec -- cargo test -p ft-kernel-cpu eigvalsh_blocked_values_matches_scalar_reference_128x128 -- --nocapture
```

Result: passed on `ovh-a`.

## Benchmark

Baseline from pass 1, same worker `ovh-a`:

- `eigh_f64_256x256`: `[8.9996 ms 9.0613 ms 9.1502 ms]`
- `eigvalsh_f64_256x256`: `[5.6935 ms 5.7728 ms 5.8773 ms]`

Candidate after row, same worker `ovh-a`:

- `eigh_f64_256x256`: `[15.042 ms 15.286 ms 15.534 ms]`
- `eigvalsh_f64_256x256`: `[6.2658 ms 6.7114 ms 7.3767 ms]`

The primary public row regressed `5.7728 ms -> 6.7114 ms`. The guard row also
drifted badly. Score: `0.0`.

## Decision

Rejected. The blocked-values source hunk and its focused candidate-only test
were removed. No source code from this candidate is shipped.

## Next route

Do not retry this single-panel delayed-top-left wedge. The next deeper primitive
should either factor a true LAPACK-style compact-WY `dsytrd` panel with `Y/W`
formation instead of reconstructing corrections around the EISPACK stream, or
profile the tridiagonal QL residual and attack the values-only solver with a
secular/divide-and-conquer primitive behind a strict fallback.
