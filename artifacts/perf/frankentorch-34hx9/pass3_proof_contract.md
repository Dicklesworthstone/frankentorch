# frankentorch-34hx9 pass 3: blocked eigvalsh reducer proof contract

## Selected lever

- Public row: `eigvalsh_f64_256x256`.
- Baseline worker: `ovh-a`.
- Baseline median: `5.7728 ms`.
- Primitive guardrail: GEMM-backed lower rank-2k update median `280.49 us`
  versus scalar `1.2644 ms` on the same worker.

The one lever for pass 4 is a values-only blocked Householder reducer for
`eigvalsh`. It must not change QL iteration, final `total_cmp` sorting, the full
`eigh` reducer/backtransform, or the rank-2k helper internals in the same commit.

## Algorithm invariant

The scalar reducer applies, for each reflector `i`, a lower rank-2 update to the
leading matrix:

```text
A00 := A00 - (v w^T + w v^T)
```

The blocked reducer may delay only the fully leading submatrix update for a
panel. It must still update the rows that will become later reflectors inside
the same panel immediately. Any dot product for a later reflector that reads a
delayed top-left entry must include the compact correction from the stored prior
`V/W` columns:

```text
A_delayed x = -V(W^T x) - W(V^T x)
```

This is the minimum safe blocked `dsytrd` wedge: panel rows stay current for
subsequent reflector generation, while the fully leading block receives the
measured BLAS-3 rank-2k update once per panel.

## Fallback and scope

- Current `eigh_tred2_values_only` remains the strict scalar reference.
- Public fast path is gated by size and finite inputs.
- Small matrices and non-finite matrices use the strict scalar path.
- Full `eigh` remains on the existing packed-full reducer.
- No RNG, randomized pivoting, or nondeterministic scheduling is introduced.

## Proof obligations

- Ordering: output remains sorted ascending by `f64::total_cmp`.
- Ties: values-only output has no eigenvector association; exact equal values
  are observable only as the sorted value sequence.
- Floating point: blocked dots and rank-2k updates reassociate arithmetic, so
  bit-exact equality to scalar is not required for the fast path. The focused
  test must compare blocked-versus-scalar eigvalsh on deterministic matrices
  with an absolute/relative tolerance ledger.
- Strict golden: `eigvalsh_golden_before.txt` SHA-256 is recorded before the
  edit, and the strict small-matrix fallback must reproduce it after the edit.
- Guards: `eigvalsh_matches_eigh`,
  `eigh_tred2_tql2_orthonormal_and_reconstructs_24x24`, and
  `symmetric_rank2k_lower_update_matches_scalar_reference` must still pass.

## Acceptance gate

Keep only if the same-worker RCH after row for `eigvalsh_f64_256x256` clears
Score `>= 2.0` and the proof gates pass. If the public-row win is noise-scale,
if the blocked/scalar tolerance proof fails, or if the fast path needs to fall
back on the benchmark matrix, remove the source hunk and record the rejection.
