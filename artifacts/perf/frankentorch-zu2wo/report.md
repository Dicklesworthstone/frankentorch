# frankentorch-zu2wo pass 1: shadow Francis column tiling

## Target

- Bead: `frankentorch-zu2wo`
- Parent route: `frankentorch-fql10` / non-symmetric Francis QR floor
- Profile-backed public hotspot baseline on `hz2`: `eigvals_f64_256x256`
  `[26.350 ms 26.900 ms 27.616 ms]`
- Lever scope: private shadow replay only; public `eig` / `eigvals` dispatch remains
  traced-disabled and unchanged.

## Lever

The shadow replay already tiled each recorded row transform. This pass tiles the
recorded column transform as well, walking `i` in the same ascending scalar order
inside fixed-size blocks. This is proof infrastructure for later grouped/small-
bulge QR work: it exercises a blocked schedule against the scalar-complete ledger
without changing production eigenvalue computation.

## Behavior proof

- Focused shadow replay tests after final formatting:
  `artifacts/perf/frankentorch-zu2wo/pass1_shadow_profile_tests_after_fmt.log`
  - Worker: `hz1`
  - Result: `3 passed; 0 failed`
- Broader eig-filter tests:
  `artifacts/perf/frankentorch-zu2wo/pass1_eig_tests.log`
  - Worker: `hz2`
  - Result: `24 passed; 0 failed`
- Strict golden stdout:
  `artifacts/perf/frankentorch-zu2wo/pass1_eigvals_golden.stdout`
  - SHA256:
    `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`
  - Digest lines stayed order-sensitive for both `eigvals` and `eig` at
    `n=64`, `n=128`, and `n=256`.
- Isomorphism ledger:
  - `EigFrancisShiftSample` stream unchanged by construction.
  - selected-`m` stream unchanged by construction.
  - active-window and deflation counters unchanged by construction.
  - final Schur buffer, eigenvalue bits, eigenvector bits, and complex-pair slot
    order covered by the hidden shadow tests and broad eig tests.
  - no RNG is introduced.

## Bench Evidence

- Baseline, `hz2`, public `eigvals_f64_256x256`:
  `artifacts/perf/frankentorch-zu2wo/pass1_baseline_eigvals_f64_256x256.log`
  - `[26.350 ms 26.900 ms 27.616 ms]`
- After, `hz2`, public `eigvals_f64_256x256`:
  `artifacts/perf/frankentorch-zu2wo/pass1_rebench_eig_hz2.log`
  - `[26.733 ms 26.978 ms 27.363 ms]`
- Supporting after attempts on `vmi1152480` are retained as smoke-only artifacts:
  - `pass1_after_eigvals_f64_256x256_hz2.log`:
    `[28.272 ms 29.836 ms 31.192 ms]`
  - `pass1_after_eigvals_f64_256x256_hz2_forced.log`:
    `[27.941 ms 28.377 ms 28.881 ms]`
- Decision: the comparable `hz2` baseline/after pair shows public dispatch did
  not move beyond Criterion noise. The source lever remains private shadow-proof
  infrastructure; no public speedup is claimed until a later grouped lane is
  bit-exact and wired.

## Gates

- `ubs crates/ft-kernel-cpu/src/lib.rs`: exit 0
- `rch exec -- cargo check -p ft-kernel-cpu --lib --examples --benches`: pass
- `rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`: pass
- `rch exec -- cargo fmt -p ft-kernel-cpu --check`: pass after line-wrap fix

## Score

- Impact: 3.0, because this completes the blocked replay scaffold needed before
  any public small-bulge/multishift QR dispatch can be safely attempted.
- Confidence: 5.0, because the scalar-complete ledger, focused shadow tests,
  broad eig tests, and strict golden SHA all pass.
- Effort: 2.0.
- Score: `3.0 * 5.0 / 2.0 = 7.50`.

Verdict: KEEP as private proof infrastructure. Successor bead
`frankentorch-bw61h` tracks the first guarded public-dispatch candidate, which
must prove bit identity against this ledger and get same-worker before/after
evidence before any keep.
