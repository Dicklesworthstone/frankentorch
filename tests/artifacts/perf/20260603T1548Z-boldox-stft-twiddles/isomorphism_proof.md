# frankentorch-nwn2 STFT DFT Twiddle Precompute

## Change

`tensor_stft` now precomputes each direct-DFT twiddle `(cos(angle), sin(angle))`
for every `(k, n)` pair once per call and reuses the table across frames. The
transform remains the existing direct DFT; this pass does not substitute a
Cooley-Tukey FFT because bit-level floating-point parity is required.

## Proof Obligations

- Ordering preserved: yes. Output layout remains `[freq_bins, frames]`, and each
  row still writes frames in ascending order.
- Tie-breaking: N/A. STFT has no ordering/tie branch.
- Floating-point: preserved for observable outputs. The twiddle table computes
  the same `angle` expression and same `cos`/`sin` values once for each `(k, n)`;
  each frame still sums samples in ascending `n` order and applies the same
  scale/cast after accumulation.
- RNG seeds: N/A. STFT has no RNG path beyond the external benchmark fixture.
- DType/shape: unchanged for F32->Complex64 and F64->Complex128 paths.
- Ledger text: unchanged.

## Validation

- `rch exec -- cargo test -p ft-api stft_parallel_match_serial_bit_exact -- --nocapture`
  passed. The filter also ran `istft_parallel_match_serial_bit_exact`; both
  passed.
- Golden fixture file: `golden_stft_outputs.txt`.
- Remote baseline: `stft/len32768_nfft512` [187.98 ms 200.34 ms 215.69 ms] on
  `vmi1156319`.
- Remote after: `stft/len32768_nfft512` [8.9232 ms 9.1057 ms 9.3781 ms] on
  `vmi1227854`.

## Score

Impact 5 x confidence 3 / effort 2 = 7.5. Keep.
