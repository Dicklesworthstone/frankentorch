# frankentorch-x9137 Pass 4 Ledger Rejection and Closeout

Date: 2026-06-13T00:55:48Z
Bead: `frankentorch-x9137`

## Result

`frankentorch-x9137` is complete as a proof-harness slice and rejected as a production-speed blocked-ledger slice.

The kept source slice is commit `17811b5b`, which added the hidden `eig_francis_shadow_profile_f64` proof harness. Public `eig_impl`, `eigvals_contiguous_f64`, and `eig_contiguous_f64` dispatch stayed unchanged.

The attempted next step, replacing the shadow replay lane with the active-window row/column ledger, did not clear the behavior gate. The replay diverged from the scalar sweep:

- values path: `80` shadow replay mismatches
- full `eig` path: `43` shadow replay mismatches
- failing evidence: `artifacts/perf/frankentorch-x9137/pass4_shadow_ledger_tests.log`

The ledger attempt therefore has Score `0.0`; no production dispatch or runtime optimization was kept.

## Baseline And Proof

Pass 1 baseline on RCH worker `hz2`:

- `eigvals_f64_256x256`: `[26.396 ms 26.500 ms 26.737 ms]`
- `eig_f64_256x256`: `[53.720 ms 55.281 ms 57.322 ms]`
- Francis profile n=256: `sweeps=319 defl1=14 defl2=121 fallback=0 exceptional=0`
- Francis profile n=1024: `sweeps=1132 defl1=18 defl2=503 fallback=0 exceptional=0`

Strict golden stdout SHA-256 stayed:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

Pass 3 proof harness validation:

- `cargo test -p ft-kernel-cpu --lib eig_francis -- --nocapture`: `4` passed on `hz2`
- `cargo test -p ft-kernel-cpu --lib eig -- --nocapture`: `24` passed on `hz2`
- `cargo check -p ft-kernel-cpu --lib --examples --benches`: passed on `vmi1227854`
- `cargo clippy -p ft-kernel-cpu --lib --examples --benches -- -D warnings`: passed locally after remote `hz2` lacked `cargo-clippy`
- `cargo fmt -p ft-kernel-cpu --check`: passed
- `ubs crates/ft-kernel-cpu/src/lib.rs`: exit `0`

Pass 3 proof-infrastructure score:

```text
Impact 3.0 * Confidence 5.0 / Effort 2.0 = 7.50
```

This is not a production speedup claim. The after benchmark selected a different worker (`vmi1227854`) and is recorded only as routing evidence.

## Isomorphism Ledger

The kept proof harness preserves:

- scalar shift source
- selected-`m` search
- active-window stream
- deflation counters
- complex-pair slot ordering
- public eigenvalue/eigenvector dispatch
- RNG absence
- strict `eigvals_golden` SHA

The rejected ledger failed before any benchmark gate, so no behavior-changing source hunk was kept for that path.

## Next Route

Open a successor bead for a scalar-complete shadow Francis sweep ledger. The next source slice must account for every scalar write in the current sweep, including subdiagonal and cleanup assignments, before attempting blocked/tiled grouping. Do not repeat range/index micro-cuts, alternate shift packets, qglh3 AED threshold variants, or public dispatch wiring before the shadow ledger is bit-exact.
