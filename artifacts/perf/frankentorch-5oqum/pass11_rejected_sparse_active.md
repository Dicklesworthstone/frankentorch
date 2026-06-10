# Pass 11 Rejected Sparse Active Reducer

## State

- Worktree: `/data/projects/frankentorch-5oqum-boldfalcon`
- HEAD: `f57499c9`
- Bead: `frankentorch-5oqum`
- Decision: rejected; no source hunk remains.

## Attempt

Implemented one private finite-only sparse/banded values-only Householder reducer for
`eigvalsh_two_stage_f64` after stage1 copied the banded lower triangle into packed
storage. The reducer skipped exact zero finite terms and was guarded to fall back to
the existing dense `eigh_tred2_values_only` for non-finite inputs, tiny shapes, or
active rows that became too dense.

The implementation was removed because the focused proof did not pass.

## Proof Commands And Workers

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p ft-kernel-cpu sparse_banded_values_only_reducer -- --nocapture`
  - Log: `pass11_sparse_active_test_remote.log`
  - Result: no remote run; RCH refused local fallback with
    `critical_pressure=1,active_project_exclusion=1`.

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p ft-kernel-cpu sparse_banded_values_only_reducer -- --nocapture`
  - Log: `pass11_sparse_active_test_remote_retry.log`
  - Worker: `vmi1227854`
  - Result: failed. Both sparse-active fixtures rejected the candidate route:
    `stage1-band n=32 b=4` and `clustered-finite-band`.

- Later retries after loosening the density guard:
  - `pass11_sparse_active_test_remote_retry2.log`: RCH refused local fallback with
    `critical_pressure=1,insufficient_slots=1`.
  - `pass11_sparse_active_test_remote_retry3.log`: remote `vmi1227854` ran zero
    matching tests, so it is not accepted as proof.
  - `pass11_sparse_active_test_remote_retry4.log`,
    `pass11_sparse_active_test_remote_retry5.log`,
    `pass11_sparse_active_test_remote_retry6.log`: RCH refused local fallback with
    `critical_pressure=1,insufficient_slots=1`.

## Benchmark

Not run for this candidate. The proof gate failed before the Criterion gate, so no
Score was computed and no same-worker before/after numbers are claimed.

## Isomorphism Summary

- Ordering: no accepted change; live and private staged outputs remain pass10 behavior.
- Tie handling: no accepted change.
- Floating point: attempted sparse path changed arithmetic by skipping exact finite
  zero multiplications, but it was removed after failing the proof gate.
- RNG: none.
- Golden SHA impact: none from this pass; source is back to pass10 behavior.

## Next Route

The sparse-active Householder route is not a keep. The useful next pass should avoid
another row/active-support micro-lever and move to a true band-to-tridiagonal
algorithmic primitive or a new measured profile target.
