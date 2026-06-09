# frankentorch-34hx9: one-GEMM finite rank-2k dsytrd prerequisite

## Target

`frankentorch-34hx9` targeted the BLAS-3 trailing-update primitive needed by blocked symmetric tridiagonalization. The measured subprimitive was:

- `sym_rank2k_lower_gemm_f64_256x32`
- worker: `vmi1227854`
- baseline: `[268.28 us 274.52 us 279.97 us]`
- after: `[181.55 us 183.67 us 185.05 us]`
- median speedup: `1.49x`
- score: `Impact 3 * Confidence 5 / Effort 2 = 7.5`

## Change

`symmetric_rank2k_lower_update_f64` now computes `V W^T` once for finite panels and uses the transposed entry for the symmetric `W V^T` term. Non-finite panels keep the old two-GEMM path.

This is one lever and does not wire the blocked DLATRD panel into `eigh_tred2_*`. That residual is tracked as `frankentorch-5oqum`.

## Isomorphism

- Ordering/tie-breaking: no public ordering path changed.
- Floating point: finite path is bit-proved against the former two-GEMM update because both terms use the same K-order dot product with finite IEEE operands; non-finite inputs keep the previous two-GEMM arithmetic.
- RNG: no RNG introduced.
- Golden SHA: extracted `eigh_golden` payload is `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`, matching the canonical checksum.

## Gates

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-kernel-cpu symmetric_rank2k_lower_update -- --nocapture`: passed on `vmi1227854`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- sym_rank2k_lower_gemm_f64_256x32 --sample-size 10 --warm-up-time 1 --measurement-time 3`: passed on `vmi1227854`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo run -p ft-kernel-cpu --example eigh_golden`: passed on `vmi1227854`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-kernel-cpu --all-targets`: passed on `vmi1227854`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`: passed on `vmi1227854`.
- `cargo fmt -p ft-kernel-cpu --check`: fails on broad pre-existing rustfmt drift outside this lever; `git diff --check` for touched paths passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: 0 critical issues; broad pre-existing warnings remain.

## Next Primitive

Attack `frankentorch-5oqum`: wire the actual DLATRD-style blocked lower panel into `eigh_tred2_reduce_packed_full` and `eigh_tred2_values_only`, then rebench `eigvalsh_f64_256x256` and `eigh_f64_256x256` on the same worker. Do not repeat scalar EISPACK loop rearrangements; that family regressed `eigh_f64_256x256` on `ovh-a` from `8.8237 ms` to `9.7077 ms`.
