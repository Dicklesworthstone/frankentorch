# frankentorch-kgs4.117 - MaxPool3d 2x2x2 stride2 backward fast path

## Profile Target

- Current routing profile: `artifacts/perf/frankentorch-next-reprofile-20260617/current_top_train_reprofile.log`
- Criterion row: `max_pool3d/grad`
- Current timing: `[35.230 ms 36.089 ms 36.992 ms]`
- Benchmark shape: `[N,C,D,H,W]=[2,32,16,32,32]`, kernel `2x2x2`, stride `2`

## Lever

Specialize `ft_kernel_cpu::max_pool3d_backward_f64` when `kd=kh=kw=2` and
`sd=sh=sw=2`.

The generic path rescans every output window with three nested kernel loops and
index arithmetic. The specialized path keeps the same first-argmax scan order
but expands the eight candidate offsets directly for the realistic volumetric
training trace.

## Correctness Guard

- Added `max_pool3d_2x2s2_backward_matches_generic_first_tie_bits`.
- The guard computes a hand-written generic reference with the previous loop
  order, including a tied first window, then compares the specialized public
  route bit-for-bit.
- Compile gate: `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo check -p ft-kernel-cpu` passed on 2026-06-18.
- Conformance and focused Criterion gates are pending by campaign instruction.

## Negative-Evidence Ledger

| Attempt | Evidence | Decision |
| --- | --- | --- |
| MaxPool3d 2x2x2 stride2 f64 backward unrolled argmax scatter | Code-first only. `cargo check -p ft-kernel-cpu` passed; focused Criterion not run by campaign instruction. No benchmark claim until focused Criterion batch. | Keep in progress for batch-test. |
| MaxPool2d borrowed-input tape plumbing | Prior ledger: `frankentorch-b03fn` regressed the 2D grad route. | Do not retry here. |
| MaxPool2d 2x2 direct/duplicate-path simplification | Prior ledgers: `frankentorch-xbvlx`, `frankentorch-3oyr5`, and `frankentorch-g047y` rejected adjacent 2D pooling micro-paths. | This bead is distinct: 3D-only, no API/tape plumbing, no 2D code touched. |

## Batch-Test Contract

Before any speed claim, run focused same-worker Criterion against
`max_pool3d/grad` and preserve or revert based on the no-gaps perf gate. If the
focused row regresses or is noise-equivalent, close this as a negative result and
do not repeat eight-lane 3D pooling unrolls without a new profile showing loop
overhead dominates.
