# frankentorch-8ubk3 Pass 4 Index-Hoist Source Rejection

Date: 2026-06-12

Scope: one source lever attempted and removed. No production source diff remains.

## Lever Tried

Inside `eig_francis_schur_traced`, the single-bulge row and column update loops
were rewritten to:

- hoist repeated `row * n` / `k * n` base-index arithmetic
- specialize the `notlast` branch outside the inner row/column loops
- preserve the same scalar shift source and selected-`m` search
- preserve the same per-slot floating-point operation order

This was an exact-shift, exact-arithmetic source slice. It did not alter public
dispatch.

## Behavior Proof

Focused eig tests passed on RCH worker `hz2`:

```text
cargo test -p ft-kernel-cpu --lib eig -- --nocapture
21 passed; 0 failed
```

Strict golden passed on RCH worker `hz2`:

```text
strict stdout sha256: 24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
n=256 eigvals_digest=0x00b87b4996340204
n=256 eig_digest=0x00b87b4996340204
```

Compiler and hygiene gates:

- `cargo check -p ft-kernel-cpu --lib --examples --benches`: PASS on RCH worker `hz1`
- `cargo clippy -p ft-kernel-cpu --lib --examples --benches -- -D warnings`:
  remote `hz1` and `hz2` lacked `cargo-clippy`; local crate-scoped fallback PASS
- `cargo fmt -p ft-kernel-cpu --check`: rch refused non-compilation command; local
  crate-scoped fallback PASS
- `ubs crates/ft-kernel-cpu/src/lib.rs`: exit 0, zero critical issues; broad
  pre-existing warning inventory remains

## Same-Worker Performance Gate

Baseline from pass 1, primary paired worker `hz1`:

```text
eigvals_f64_256x256 [33.839 ms 34.014 ms 34.212 ms]
eig_f64_256x256     [68.327 ms 68.927 ms 69.570 ms]
```

Candidate after row on the same worker `hz1`:

```text
eigvals_f64_256x256 [39.920 ms 41.337 ms 42.774 ms]
```

Median ratio:

```text
34.014 / 41.337 = 0.82x
```

The primary `eigvals` row regressed, so the source hunk failed the Score gate.
The full `eig` after row was not run because a keep was already impossible.

## Decision

Rejected. The source hunk was removed and `crates/ft-kernel-cpu/src/lib.rs`
returned to no diff.

Score:

```text
0.0
```

## Next Route

Do not repeat index-hoist, branch-specialization, row-range, or alternate-shift
micro-levers. Pass 5 should route deeper to a fundamentally different exact
fallback primitive:

```text
shadow active-window blocked sweep kernel
```

The candidate should clone one active Hessenberg window, run a blocked/tiled
application of the already-selected scalar reflector sequence in shadow, compare
the resulting window and exact stream against the scalar path, and only then
consider guarded production dispatch. This attacks the blocked-update structure
without changing shift policy.
