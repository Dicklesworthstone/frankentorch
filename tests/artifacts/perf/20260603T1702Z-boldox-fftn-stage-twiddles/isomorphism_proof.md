# frankentorch-rdem FFT Stage Twiddle Reuse

## Change

`tensor_fft_along_dim` now builds the Cooley-Tukey stage twiddle table once for
same-size power-of-two lanes and passes that table into each lane transform.
Standalone `dft_inplace_1d` keeps its previous inline table path, while the
shared helper supports the reused-table path for batched FFTN lanes.

This is still the existing iterative FFT algorithm. The lever removes repeated
twiddle construction across lanes; it does not change FFT complexity, stage
ordering, lane ordering, or butterfly arithmetic.

## Proof Obligations

- Ordering preserved: yes. `tensor_fftn` still applies dimensions in the same
  requested order. Within each lane, bit reversal, stage order, start block
  order, and `k` butterfly order are unchanged.
- Tie-breaking: N/A. FFT has no compare/tie branch.
- Floating-point: preserved for observable outputs. Shared twiddles use the
  same `sign * 2*pi / stage` angle step and the same `angle_step * k`
  expression as the inline path. Butterfly multiply/add/subtract order is
  unchanged.
- RNG seeds: N/A for the operation. Criterion uses the existing benchmark
  fixture generator.
- DType/shape: unchanged. Real input still returns Complex128 output through
  `tensor_complex`; complex input keeps the same real/imag extraction path.
- Parallelism: unchanged. Lanes remain independent Rayon chunks; only the
  read-only twiddle slice is shared.
- Fallback: non-power-of-two transforms still use the naive DFT path.

## Validation

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api dft_stage_twiddles_match_per_butterfly_reference_bit_exact -- --nocapture`
  passed on `vmi1167313`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api fft_along_dim_parallel_match_serial_bit_exact -- --nocapture`
  passed on `vmi1227854`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-api --all-targets` passed remotely on
  `vmi1153651`.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-api --all-targets -- -D warnings` failed
  before this diff in `ft-kernel-cpu` SVD lints at
  `crates/ft-kernel-cpu/src/lib.rs:5725`, `5744`, `5793`, `5948`, and `5985`.
- `cargo fmt -p ft-api --check` failed on pre-existing broad ft-api formatting
  drift across benches and source. `git diff --check -- crates/ft-api/src/lib.rs`
  passed for this owned code diff.
- Golden fixture file: `golden_fftn_outputs.txt`.
- Same-worker remote baseline: `fftn/512x2048_dim1`
  `[22.417 ms 26.137 ms 29.363 ms]` on `vmi1293453`.
- Same-worker remote after: `fftn/512x2048_dim1`
  `[20.458 ms 21.580 ms 22.191 ms]` on `vmi1293453`.

## Score

Impact 2 x confidence 4 / effort 2 = 4.0. Keep.
