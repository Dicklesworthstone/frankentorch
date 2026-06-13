# frankentorch-gxpb2 pass 3 const WANT_VECTORS keep

Scope: production `eigvals` values-only specialization for the existing Francis Schur core.

Source lever: commit `abd1f2d2` monomorphizes `eig_francis_schur_traced` over
`const WANT_VECTORS: bool`. The wrapper still dispatches from the same
`want_vectors` boolean, but the hot Schur loop now has separate values-only and
full-vector instantiations. The values-only path lets LLVM remove Schur-vector
standardization and `q_acc` accumulation branches instead of carrying them
through each bulge-chase iteration.

Baseline and rebench:

- Baseline: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench eigvals_f64_256x256 -- --warm-up-time 1 --measurement-time 3 --sample-size 10`
- Worker: `vmi1227854`
- Before: `eigvals_f64_256x256` `[25.340 ms 26.502 ms 27.201 ms]`
- Candidate: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench eigvals_f64_256x256 -- --warm-up-time 1 --measurement-time 3 --sample-size 10`
- Worker: `vmi1227854`
- After: `eigvals_f64_256x256` `[24.206 ms 24.456 ms 24.627 ms]`
- Median speedup: `1.084x`

Behavior proof:

- Ordering and complex-pair slots: unchanged. The eigenvalue-producing arithmetic
  on `h`, selected-m search, deflation order, and fallback/exceptional-shift
  policy are identical; only the vector-side branch is lifted to compile time.
- Floating point: no reassociation in the values path. The same row/column update
  loops write the same `h` slots in the same order.
- RNG: none.
- Strict golden: `eigvals_golden` stdout SHA-256 stayed
  `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
- Focused tests: `cargo test -j 1 -p ft-kernel-cpu --lib eig_francis -- --nocapture`
  passed `4/4`.
- Broad eig filter: `cargo test -j 1 -p ft-kernel-cpu --lib eig -- --nocapture`
  passed `24/24` on `vmi1227854`.

Quality gates:

- `cargo check -j 1 -p ft-kernel-cpu --all-targets` passed on `hz1`.
- `cargo clippy -j 1 -p ft-kernel-cpu --all-targets -- -D warnings` passed on `vmi1149989`.
- `cargo fmt -p ft-kernel-cpu --check` passed locally.
- `ubs crates/ft-kernel-cpu/src/lib.rs` exited `0`; it reported no critical issues
  and only pre-existing broad warnings in the large kernel file.

Score:

- Impact `3.0` (8.4% median win on the shared `eigvals` QR floor)
- Confidence `4.0` (same-worker Criterion + strict golden + focused and broad tests)
- Effort `2.0`
- Score `6.0`

Verdict: KEEP. `frankentorch-gxpb2` remains in progress for the actual
size-gated AED/multishift dispatch. The next source pass should attack a deeper
grouped/far-update or strict Schur-window primitive rather than another range or
shift-policy micro-lever.
