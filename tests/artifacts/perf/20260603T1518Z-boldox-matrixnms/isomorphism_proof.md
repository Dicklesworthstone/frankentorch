# frankentorch-kgs4.33 Matrix NMS Bitset Precompute

## Change

`matrix_nms` now thresholds each selected mask once into packed `u64` words and
precomputes each mask area once. Pairwise IoU then uses `AND + popcount` instead
of re-reading and re-thresholding both masks for every pair.

## Proof Obligations

- Ordering preserved: yes. The score sort, top-k truncation, row order, decay
  loop, final score sort, and returned index order are unchanged.
- Tie-breaking unchanged: yes. Sort comparators are unchanged and this lever
  does not add any new ordering rule.
- Floating-point: unchanged for observable scores. The old mask loop only summed
  `0.0` or `1.0` threshold indicators; the new integer counts are exactly
  representable in `f64` for this path before the same `max(1e-6)`, division,
  `exp`, decay multiplication, and score multiplication.
- RNG seeds: N/A. Matrix NMS has no RNG.
- Golden outputs: `golden_matrix_nms_outputs.txt` records the representative
  `matrix_nms_parallel_match_serial_bit_exact` output indices and score bits.

## Validation

- `rch exec -- cargo test -p ft-api matrix_nms_parallel_match_serial_bit_exact -- --nocapture`
  passed: 1 test passed, 1812 filtered.
- Baseline Criterion: `matrix_nms/256x48x48` [57.077 ms 86.618 ms 127.85 ms]
  on `vmi1153651`.
- After Criterion: `matrix_nms/256x48x48` [4.0367 ms 4.1755 ms 4.3428 ms]
  on `vmi1293453`.

## Score

Impact 5 x confidence 3 / effort 2 = 7.5. Keep.
