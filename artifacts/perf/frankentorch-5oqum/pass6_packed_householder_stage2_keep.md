# frankentorch-5oqum pass 6: packed-Householder stage-2 keep

## Target

Profile-backed target: `eigvalsh_two_stage_f64` remained slower than the live
packed-Householder `eigvalsh_contiguous_f64` after the BLAS-3 stage-1 commit.
Same-worker current-HEAD baselines on `vmi1227854`:

- `eigvalsh_f64_128x128`: `[1.1065 ms 1.1357 ms 1.1689 ms]`
- `eigvalsh_two_stage_f64_128x128_b16`: `[2.7803 ms 2.8136 ms 2.8504 ms]`
- `eigvalsh_f64_256x256`: `[7.1906 ms 7.8459 ms 8.8382 ms]`
- `eigvalsh_two_stage_f64_256x256_b32`: `[29.165 ms 30.240 ms 32.288 ms]`
- `banded_to_tridiag_f64_256x256_b32`: `[22.056 ms 22.985 ms 23.855 ms]`

This rejected a live dispatch swap and identified stage 2 as the next bottleneck.

## Lever

One lever: replace the staged two-stage path's Givens bulge-chase stage 2 with
the existing values-only packed Householder tridiagonalization over the banded
matrix's lower triangle. Public `eigvalsh_contiguous_f64` and `eigh_contiguous_f64`
dispatch remain unchanged.

Isomorphism contract:

- Input band is copied lower-triangular into packed storage with row-major order
  preserved inside each lower row.
- `eigh_tred2_values_only` and `eigh_tql2_values_only` preserve the existing
  values-only ordering and `total_cmp` final sort.
- No RNG, no tie-breaking changes, no public dispatch change.

## Results

Same-worker rebench on `vmi1227854`:

- `eigvalsh_two_stage_f64_128x128_b16`: `2.8136 ms -> 1.7038 ms` median,
  `1.65x`.
- `eigvalsh_two_stage_f64_256x256_b32`: `30.240 ms -> 12.100 ms` median,
  `2.50x`.

The staged path is still slower than live dispatch at these sizes (`1.1357 ms`
and `7.8459 ms` medians respectively), so no live swap was made.

Score: Impact 3 x Confidence 4 / Effort 2 = 6.0. Keep.

## Proof

- `cargo test -j 1 -p ft-kernel-cpu eigvalsh_two_stage_matches_live -- --nocapture`
  on `vmi1227854`: pass.
- `FT_EIGVALSH_GOLDEN=1 cargo run -j 1 -p ft-kernel-cpu --example eigh_golden`
  on `vmi1227854`: SHA `1870e56ea935f9cc895b24d878db52fe341dc2b195c00656faa38b2db97ac458`,
  unchanged from pass 3.
- `cargo check -j 1 -p ft-kernel-cpu` on `vmi1227854`: pass.
- `cargo clippy -j 1 -p ft-kernel-cpu -- -D warnings` on `vmi1227854`: pass.
- `cargo fmt -p ft-kernel-cpu --check`: pass.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: exit 0, no critical findings; broad
  pre-existing warning inventory remains.

## Next Route

Do not claim the project is at a ceiling. The live gap remains:

- 128x128: staged `1.7038 ms` vs live `1.1357 ms`.
- 256x256: staged `12.100 ms` vs live `7.8459 ms`.

The next algorithmic primitive should attack stage 2 again, but not by
micro-tuning the old full-matrix bulge-chase loop. Target a true compact-band
or divide-and-conquer / MRRR-style values-only symmetric band eigensolver with
native safe-Rust storage.
