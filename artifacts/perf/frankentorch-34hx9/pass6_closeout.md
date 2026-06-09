# frankentorch-34hx9 pass 6: closeout and handoff

## Outcome

The first public-path blocked-values wedge for `frankentorch-34hx9` was rejected
and removed. No source code from the candidate is shipped.

## Evidence bundle

- Pass 1 baseline/profile:
  - `eigh_f64_256x256`: `[8.9996 ms 9.0613 ms 9.1502 ms]`
  - `eigvalsh_f64_256x256`: `[5.6935 ms 5.7728 ms 5.8773 ms]`
  - `sym_rank2k_lower_scalar_f64_256x32`: `[1.2448 ms 1.2644 ms 1.2860 ms]`
  - `sym_rank2k_lower_gemm_f64_256x32`: `[266.05 us 280.49 us 300.34 us]`
- Candidate after:
  - `eigh_f64_256x256`: `[15.042 ms 15.286 ms 15.534 ms]`
  - `eigvalsh_f64_256x256`: `[6.2658 ms 6.7114 ms 7.3767 ms]`
- Scalar recheck after removing candidate:
  - `eigh_f64_256x256`: `[9.1715 ms 9.2351 ms 9.3055 ms]`
  - `eigvalsh_f64_256x256`: `[5.7113 ms 6.2999 ms 7.2085 ms]`

## Proof bundle

- Candidate-only scalar-vs-blocked 128x128 proof passed before benchmarking.
- Post-rejection `eigvalsh_matches_eigh` passed.
- Strict eigvalsh golden SHA before and after rejection:
  `1870e56ea935f9cc895b24d878db52fe341dc2b195c00656faa38b2db97ac458`.

## Bead status

`frankentorch-34hx9` remains `in_progress`. The bead is not complete because the
blocked `dsytrd` no-gaps target has not shipped.

The bead notes were updated with:

- rejected candidate family
- exact before/candidate/scalar-recheck medians
- unchanged strict golden SHA
- next route

## Handoff

Do not retry the delayed-top-left EISPACK wedge. Next attack should be a true
compact-WY `dsytrd` panel with explicit `Y/W` formation, or a fresh split
profile proving that tridiagonal QL is now the next residual before attempting a
values-only secular/divide-and-conquer solver.
