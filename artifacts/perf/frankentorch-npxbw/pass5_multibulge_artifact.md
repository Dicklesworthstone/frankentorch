# Pass 5 - Direct Two-Bulge Multishift Artifact

Date: 2026-06-11
Bead: `frankentorch-npxbw`

## Trigger

Pass 4 rejected an eigvals-only range micro-cut. Per the no-ceiling directive,
the next move is not another local loop trim. The next pass must replace the
one-bulge scalar sweep family with a structurally different direct multibulge
primitive.

## Profile Refresh

Command:

```bash
RCH_WORKER=vmi1227854 RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never \
  rch exec -- cargo run --release -q -j 1 -p ft-kernel-cpu --example eig_timing_probe \
  > artifacts/perf/frankentorch-npxbw/pass5_multibulge_profile_eig_timing_probe.log 2>&1
```

Remote worker: `vmi1227854`

| n | eigvals | eig | sweeps | defl1 | defl2 | fallback | exceptional | max width |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | `3.50 ms` | `6.89 ms` | 173 | 28 | 50 | 0 | 0 | 128 |
| 256 | `30.17 ms` | `47.73 ms` | 319 | 14 | 121 | 0 | 0 | 256 |
| 512 | `368.93 ms` | `493.08 ms` | 583 | 10 | 251 | 0 | 0 | 512 |
| 1024 | `2702.55 ms` | `4360.75 ms` | 1132 | 18 | 503 | 0 | 0 | 1024 |

Interpretation: this benchmark fixture is not recovery-bound. There are no
exceptional shifts and no fallbacks. The available win is sweep count and
row/column update locality, not threshold tuning.

## Alien Primitive Mapping

Graveyard section 9.6, Communication-Avoiding Algorithms, is the right mapping:
reduce data movement and communication rounds by processing a block/panel of
linear-algebra work, then apply dense trailing updates with cache-local kernels.
For Hessenberg QR, the corresponding primitive is not full CAQR; the matrix is
already Hessenberg. It is LAPACK-style small-bulge multishift QR: introduce a
packet of shifts, chase multiple bulges through the active window, and batch the
far row/column updates once the bulges are separated.

## Pass 6 Source Slice

Implement only a private eigvals-only pilot, behind the current scalar fallback:

1. Add a private `FrancisShiftPair { x, y, w }` and `FrancisShiftPacket4`.
2. Build the packet from two current-compatible double-shift sources:
   - primary pair: the existing trailing `(en-1,en)` source already recorded as
     `(x, y, w)`;
   - secondary pair: the adjacent active `(en-3,en-2)` 2x2 source when
     `en >= l + 3`; otherwise reject to scalar.
3. Gate the pilot to `want_vectors == false`, `en + 1 - l >= 96`, no exceptional
   shift, no fallback state, and finite packet scalars.
4. Introduce two small bulges separated by two rows, but keep near-bulge updates
   scalar and in the same row-major order inside each local 3-row packet.
5. Do not batch far updates in the first source pass. The first source pass must
   prove the shift packet and two-bulge chase can preserve strict goldens before
   compact-WY/BLAS-3 batching is attempted.

This is intentionally narrower than full `dlaqr0`: one private eigvals pilot,
same output contract, scalar fallback for every unsupported state.

## Proof Obligations

- Ordering: retain bottom-up `en` deflation and existing eigenvalue slot writes.
- Complex-pair convention: upper slot positive imaginary part, lower slot
  negative imaginary part.
- Tie-breaking: existing `eps * s` split tests, `its == 10 || its == 20`
  exceptional cadence, and `max_total = 60 * n + 100` remain scalar fallback
  gates.
- Floating point: no far-update batching in pass 6; local bulge operations keep
  deterministic row-major arithmetic order inside each touched row/column.
- RNG: none.
- Golden: strict `eigvals_golden` SHA must remain
  `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
- Iteration safety: reject if `total_sweeps` increases for the n=256 golden
  fixture, if fallback count becomes nonzero, or if exceptional shifts appear.

## Keep Gate

Compare against the pass-3/pass-4 current same-worker row:

| Row | Current |
| --- | --- |
| `eigvals_f64_256x256` | `[25.052 ms 25.370 ms 25.694 ms]` |

Keep only if the pass-6 source lever:

- preserves strict golden SHA and focused eig tests,
- passes crate-scoped `check`, `clippy -D warnings`, `fmt --check`, and UBS,
- improves `eigvals_f64_256x256` on the same worker with non-overlapping or
  clearly favorable Criterion intervals,
- scores at least `Impact 5 * Confidence 3 / Effort 4 = 3.75`.

If the first two-bulge pilot changes output or regresses, reject it and route to
an AED-derived shift-list artifact, not another range cut.
