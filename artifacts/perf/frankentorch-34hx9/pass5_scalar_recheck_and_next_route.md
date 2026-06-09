# frankentorch-34hx9 pass 5: scalar recheck and deeper route

## Scalar recheck

After removing the rejected blocked-values source hunk, the public pair was
rerun on `ovh-a`:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- 'eigh_f64_256x256|eigvalsh_f64_256x256' --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Rows:

- `eigh_f64_256x256`: `[9.1715 ms 9.2351 ms 9.3055 ms]`
- `eigvalsh_f64_256x256`: `[5.7113 ms 6.2999 ms 7.2085 ms]`

The `eigh` guard returned to the pass-1 neighborhood. `eigvalsh` had high
outliers, but the immediate scalar median `6.2999 ms` was still faster than the
candidate median `6.7114 ms`, confirming the candidate did not clear the
same-worker keep gate.

## Post-rejection proof

- `cargo test -p ft-kernel-cpu eigvalsh_matches_eigh -- --nocapture` passed.
  RCH selected `vmi1227854` for this correctness-only guard.
- Post-rejection strict golden was generated on `ovh-a`.
- `cmp eigvalsh_golden_before.txt eigvalsh_golden_after_rejection.txt` passed.
- SHA-256 before and after:
  `1870e56ea935f9cc895b24d878db52fe341dc2b195c00656faa38b2db97ac458`.

## Next route

Keep `frankentorch-34hx9` open. The next pass should not retry the
delayed-top-left EISPACK wedge. Attack one of these deeper primitives:

1. True compact-WY `dsytrd` panel formation with explicit `Y/W` updates, not
   corrections wrapped around scalar `tred2`.
2. Fresh split profile of tridiagonalization versus values-only QL, followed by
   a tridiagonal divide-and-conquer/secular values solver if QL is now the
   measured residual.

Both routes need strict scalar fallback, golden SHA, and same-worker public-row
Score `>= 2.0`.
