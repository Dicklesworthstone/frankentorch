# frankentorch-npxbw Pass 3 Shift-Trace Scaffold Report

Date: 2026-06-11
Agent: IvoryDeer
HEAD: `4fc3b382`

## Change

Implemented one private observational scaffold around `eig_francis_schur` in
`crates/ft-kernel-cpu/src/lib.rs`:

- `EigFrancisShiftSample`: `#[doc(hidden)]` diagnostic packet for active window, iteration, accumulated
  shift, current double-shift scalars `(x, y, w)`, and exceptional-shift flag.
- `FrancisTraceSink`: private sink trait with a `FrancisTraceDisabled` production
  sink.
- `EigFrancisProfile`: `#[doc(hidden)]` diagnostic sweep trace recording active windows,
  selected `m`, sweep count, 1x1/2x2/fallback deflations, exceptional shifts, and
  max-total exhaustion count.
- `eig_francis_profile_f64`: `#[doc(hidden)]` diagnostic proof/profiling helper.

Production `eig_contiguous_f64` and `eigvals_contiguous_f64` still call
`eig_francis_schur`, which instantiates `FrancisTraceDisabled`. The trace calls are
guarded by the compile-time `T::ENABLED` associated const, and no trace data feeds
back into Hessenberg entries, Schur-vector replay, deflation decisions, or returned
eigenvalues. The hidden helper exists so crate-local examples can print shift-source
profiles without changing production dispatch.

## Isomorphism Proof

- Ordering preserved: bottom-up `en` loop, `l` search, selected `m`, and 1x1/2x2
  eigenvalue slot writes are unchanged.
- Tie-breaking unchanged: the existing `eps * s` split tests, exceptional shift
  cadence, and `max_total = 60 * n + 100` guard are unchanged.
- Floating-point identical: trace packets copy already-computed scalars only; they
  do not alter arithmetic order or branch predicates.
- RNG unchanged: none in this path.
- Public dispatch behavior unchanged: the concrete profile/result/helper are
  `#[doc(hidden)]` diagnostics only; production eig/eigvals APIs and outputs are
  unchanged.

## Verification

Focused RCH tests:

```bash
RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
  rch exec -- cargo test -p ft-kernel-cpu --lib eig -- --nocapture \
  > artifacts/perf/frankentorch-npxbw/pass3_focused_eig_tests_final.log 2>&1
```

Result: PASS on remote `vmi1227854`.

- `eigvals_matches_eig`: ok
- `eigvals_companion_complex_roots`: ok
- `eig_parallel_schur_vector_update_matches_single_thread_bit_exact`: ok
- `eig_francis_profile_matches_eigvals_bit_exact`: ok
- Summary: `21 passed; 0 failed; 425 filtered out`

Golden regeneration:

```bash
RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
  rch exec -- cargo run -p ft-kernel-cpu --example eigvals_golden \
  > artifacts/perf/frankentorch-npxbw/pass3_eigvals_golden.stdout \
  2> artifacts/perf/frankentorch-npxbw/pass3_eigvals_golden.stderr

rg --no-filename \
  "^(frankentorch-l9xod eigvals_golden|eigvals_digest=|eig_digest=)" \
  artifacts/perf/frankentorch-npxbw/pass3_eigvals_golden.stdout \
  artifacts/perf/frankentorch-npxbw/pass3_eigvals_golden.stderr \
  > artifacts/perf/frankentorch-npxbw/pass3_eigvals_golden.strict.stdout

sha256sum artifacts/perf/frankentorch-npxbw/pass3_eigvals_golden.strict.stdout \
  > artifacts/perf/frankentorch-npxbw/pass3_eigvals_golden.strict.stdout.sha256

diff -u \
  artifacts/perf/frankentorch-npxbw/pass1_eigvals_golden.strict.stdout \
  artifacts/perf/frankentorch-npxbw/pass3_eigvals_golden.strict.stdout \
  > artifacts/perf/frankentorch-npxbw/pass3_eigvals_golden.strict.diff
```

Result: PASS on remote `vmi1227854`.

- Strict SHA: `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`
- Strict diff size: `0` bytes

Crate check:

```bash
RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
  rch exec -- cargo check -p ft-kernel-cpu --lib --examples --benches \
  > artifacts/perf/frankentorch-npxbw/pass3_check_ft_kernel_cpu_lib_examples_benches.log 2>&1
```

Result: BLOCKED for the intermediate test-only helper version.
`crates/ft-kernel-cpu/examples/eig_timing_probe.rs` has an uncommitted peer hunk
importing `ft_kernel_cpu::eig_francis_profile_f64` as a crate item, so the final
tree exposes the diagnostic helper as `#[doc(hidden)] pub`.

Narrower crate check excluding the peer-modified example surface:

```bash
RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
  rch exec -- cargo check -p ft-kernel-cpu --lib --benches \
  > artifacts/perf/frankentorch-npxbw/pass3_check_ft_kernel_cpu_lib_benches.log 2>&1
```

Result: PASS. RCH executed on remote `vmi1149989`; `Finished dev profile`.

Formatting:

```bash
cargo fmt -p ft-kernel-cpu --check \
  > artifacts/perf/frankentorch-npxbw/pass3_fmt_check_ft_kernel_cpu.log 2>&1
```

Result: PASS.

UBS:

```bash
ubs crates/ft-kernel-cpu/src/lib.rs \
  > artifacts/perf/frankentorch-npxbw/pass3_ubs_ft_kernel_cpu_lib.log 2>&1
```

Result: exit 0. UBS reported `0` critical issues; it also reported existing
warning/info inventories in the large file.

## Blockers And Notes

- Final current-tree `--examples` check passes with the hidden diagnostic helper.
  I did not edit or revert the peer-owned `eig_timing_probe.rs` hunk.
- No qglh3 AED suffix, whole-window threshold AED, symmetric eigvalsh/eigh, x53r3,
  public multibulge chase, or BLAS-3 far update was implemented.

## Final Current-Tree Gates

- `RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-kernel-cpu --lib --examples --benches`
  PASS, log `pass3_check_current4_lib_examples_benches.log`.
- `RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-kernel-cpu --lib --examples --benches -- -D warnings`
  PASS, log `pass3_clippy_current3_lib_examples_benches.log`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu --lib eig -- --nocapture`
  PASS, `21 passed; 0 failed; 425 filtered out`, log `pass3_focused_eig_tests_current.log`.
- `RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 rch exec -- cargo run -p ft-kernel-cpu --example eigvals_golden`
  PASS, strict SHA `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`,
  empty strict diff.
- `cargo fmt -p ft-kernel-cpu --check` PASS, log `pass3_fmt_check_current2_after_lintfix.log`.
- `ubs crates/ft-kernel-cpu/src/lib.rs` PASS with `0` critical issues, log
  `pass3_ubs_ft_kernel_cpu_lib_current.log`.

## Rebench

Same-worker `vmi1227854` Criterion row:

| Row | Pass 1 baseline | Pass 3 current |
| --- | --- | --- |
| `eigvals_f64_256x256` | `[24.799 ms 25.258 ms 25.738 ms]` | `[25.052 ms 25.370 ms 25.694 ms]` |

Verdict: neutral observational scaffold; behavior proof and future multishift
source visibility are the value. Runtime speedup is intentionally deferred to
Pass 4.
