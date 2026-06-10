# frankentorch-5oqum pass 9: cap values-only SBR panel width at 16

## Target

Profile-backed staged eigvalsh residual after pass 6 on RCH worker `vmi1227854`:

- `eigvalsh_two_stage_f64_256x256_b32`: `[11.904 ms 12.100 ms 12.232 ms]`
- Stage-1 values-only harness, `n=256 b=32`: `5135.96 us/iter`
- Stage-1 values-only harness, `n=512 b=32`: `42120.21 us/iter`

Public live dispatch was still faster at `eigvalsh_f64_256x256` median `7.8459 ms`, so this pass keeps the staged experimental path only.

## Lever

One lever: keep target half-bandwidth `b` unchanged, but cap the communication-avoiding values-only SBR panel width at `16`:

- before: panel width `w = b.min(ncols - s)`
- after: panel width `w = b.min(16).min(ncols - s)`

This preserves the same band target, no RNG, same `total_cmp` sort, and the same public eigvalsh/eigh dispatch. It only changes floating-point association inside the staged values-only two-stage helper.

## Results

Same-worker `vmi1227854` stage harness:

- `n=256 b=32`: `5135.96 us/iter -> 4118.97 us/iter` (`1.25x`)
- `n=512 b=32`: `42120.21 us/iter -> 26677.49 us/iter` (`1.58x`)

Same-worker Criterion:

- `eigvalsh_two_stage_f64_256x256_b32`: `12.100 ms -> 8.9436 ms` median (`1.35x`)

Score: Impact 4 * Confidence 4 / Effort 2 = 8.0. Keep.

## Proof

- `cargo test -j 1 -p ft-kernel-cpu symmetric_to_banded_values_matches_unblocked -- --nocapture` on `vmi1227854`: pass.
- `cargo test -j 1 -p ft-kernel-cpu eigvalsh_two_stage_matches_live -- --nocapture` on `vmi1227854`: pass.
- `FT_EIGVALSH_GOLDEN=1 cargo run -j 1 -p ft-kernel-cpu --example eigh_golden` on `vmi1227854`: SHA `1870e56ea935f9cc895b24d878db52fe341dc2b195c00656faa38b2db97ac458`, unchanged from pass 6.
- `cargo check -j 1 -p ft-kernel-cpu` on `vmi1227854`: pass.
- `cargo clippy -j 1 -p ft-kernel-cpu -- -D warnings` on `vmi1227854`: pass.
- `cargo fmt -p ft-kernel-cpu --check`: pass.
- `git diff --check`: pass.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: exit 0, no critical findings; broad pre-existing warning inventory remains.

## Next Route

The staged path improved but is not yet a public swap: staged `8.9436 ms` vs live `7.8459 ms` at 256x256. Continue with a deeper values-only symmetric band/tridiagonal primitive or a true DLATRD-style stage-1 rank-2k update. Do not spend another pass on allocation reuse or naive compact storage; pass 7 and pass 8 rejected those routes.
