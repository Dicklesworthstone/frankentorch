# frankentorch-ncwz pass 1: full-vector eigh reflector cache

Target: `ft-kernel-cpu` full-vector `eigh_f64_256x256`.

Profile-backed baseline:

- Command: `rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigh_f64_256x256`
- Worker: `ts2`
- Before median: `77.522 ms`
- Adjacent split evidence: `eigvalsh_f64_256x256` median `11.889 ms`, confirming vector accumulation dominates full `eigh`.

Lever:

- Cache the active Householder reflector column during `eigh_tred2` eigenvector back-transform.
- Preserve the original `j` then `k` arithmetic order, tie behavior, floating-point operations, and eigenvector ordering.

Behavior proof:

- Golden command: `rch exec -- cargo run -p ft-kernel-cpu --example eigh_golden --quiet`
- Before SHA-256: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- After SHA-256: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- `cmp artifacts/perf/frankentorch-ncwz/eigh_golden_before.txt artifacts/perf/frankentorch-ncwz/eigh_golden_after.txt` passed.

After benchmark:

- Command: `rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- eigh_f64_256x256`
- Worker: `ts2`
- After median: `74.926 ms`
- Delta: `3.35%` faster, `1.0346x`.
- Score: `Impact 1.3 x Confidence 5 / Effort 2 = 3.25`, keep.

Verification:

- `rch exec -- cargo check -p ft-kernel-cpu --all-targets`
- `rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`
- `rch exec -- cargo test -p ft-kernel-cpu`
- `ubs crates/ft-kernel-cpu/src/lib.rs crates/ft-kernel-cpu/examples/eigh_golden.rs .skill-loop-progress.md`

Residual:

- The next deeper primitive is tridiagonal divide-and-conquer for full-vector `eigh`, because `tql2` rotation accumulation and Householder back-transform remain the dominant full-vector path.
