# ft-api Matrix NMS Packed Bitsets Supplemental Validation

- Bead: `frankentorch-kgs4.33`
- Parent umbrella: `frankentorch-kgs4`
- Skills: `/repeatedly-apply-skill`, `/extreme-software-optimization`, `/alien-graveyard`, `/alien-artifact-coding`
- Crate: `ft-api`
- Target benchmark: `ops_bench/matrix_nms/256x48x48`
- Outcome: supplemental validation after kept commit `ef8486fc`

## Profile Target

`matrix_nms` computes a selected-mask IoU matrix over `K x K` mask pairs. The
previous row-parallel implementation still thresholded every mask pair inside
the pairwise loop, so it scanned each mask many times.

Baseline from the bead, measured with rch on worker `vmi1153651`:

```text
matrix_nms/256x48x48    time: [57.077 ms 86.618 ms 127.85 ms]
```

## Lever

Precompute every selected mask once as packed `u64` words plus its exact positive
area. Pairwise IoU then computes intersections with bitwise `AND` and
`count_ones`, converting the exact integer counts to `f64` before the unchanged
decay math.

Alien primitive: succinct bitsets / SWAR popcount plus cache-local packed mask
layout.

## Isomorphism

- Ordering: score sort order, selected-index order, row order, decay order, and
  final score sort are unchanged.
- Tie-breaking: the same stable sort comparator remains in place for both score
  sorts.
- Floating point: thresholded mask indicators are counted as exact integers, then
  converted to `f64`; decay multiplication and `exp` calls remain in the same
  order as before.
- RNG: no RNG path is present.
- Golden output: `artifacts/optimization/golden_outputs/ft_api_matrix_nms_bitsets_frankentorch-kgs4-33.txt`
  records the representative output indices and score bit patterns.

## Result

Supplemental after run with rch on worker `vmi1156319`:

```text
matrix_nms/256x48x48    time: [7.4968 ms 7.8288 ms 8.1683 ms]
```

Delta by p50: `86.618 ms -> 7.8288 ms`, about `11.06x` faster.

Score for this follow-up run: Impact 4 x Confidence 2 / Effort 1 = 8.0.
