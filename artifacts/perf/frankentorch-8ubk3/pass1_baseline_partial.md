# frankentorch-8ubk3 Pass 1 Partial Baseline

Date: 2026-06-12

Scope: initial baseline for successor bead `frankentorch-8ubk3`.

## Completed Row

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigvals_f64_256x256
worker: ovh-a
eigvals_f64_256x256 time: [23.550 ms 23.596 ms 23.646 ms]
```

The first attempt in `pass1_bench_eigvals_256.log` reached benchmark startup
but did not contain a measured row; the appended rerun above is the usable
baseline.

## Remaining Pass-1 Gates

- `eig_f64_256x256` Criterion row
- strict `eigvals_golden` SHA
- `eig_timing_probe` profile counters for n=256 and n=1024

No source files were edited.
