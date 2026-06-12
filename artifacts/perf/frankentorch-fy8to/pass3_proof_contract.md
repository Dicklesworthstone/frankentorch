# frankentorch-fy8to Pass 3 Proof Contract

Date: 2026-06-12

Scope: proof-contract artifact only for bead `frankentorch-fy8to`.
No production source was edited in this pass. This contract gates the next
source pass: a strict-fallback Schur-window / AED-derived shift-list consumed
by the existing scalar Francis single-bulge chase for `geev`/`eigvals`.

## Inputs

Baseline/profile artifact: `artifacts/perf/frankentorch-fy8to/pass1_baseline_profile.md`.

Primitive-selection artifact:
`artifacts/perf/frankentorch-fy8to/pass2_alien_primitive_refinement.md`.

Current strict golden SHA-256:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

Profile-backed target:

| Evidence | Worker | Row / fixture | Current result |
| --- | --- | --- | --- |
| Criterion | `vmi1149989` | `eigvals_f64_256x256` | `[33.768 ms 35.625 ms 37.476 ms]` |
| Criterion | `vmi1149989` | `eig_f64_256x256` | `[50.173 ms 51.101 ms 52.098 ms]` |
| Supplemental Criterion | `vmi1227854` | `eigvals_f64_256x256` | `[24.692 ms 25.049 ms 25.415 ms]` |
| Supplemental Criterion | `vmi1227854` | `eig_f64_256x256` | `[73.003 ms 75.979 ms 79.258 ms]` |
| Profile | `vmi1227854` | n=256 | `sweeps=319`, `defl1=14`, `defl2=121`, `fallback=0`, `exceptional=0` |
| Profile | `vmi1227854` | n=1024 | `sweeps=1132`, `defl1=18`, `defl2=503`, `fallback=0`, `exceptional=0` |

Source anchors:

- `eig_impl` reduces the input to upper Hessenberg form, then calls
  `eig_francis_schur`.
- `eig_francis_schur_traced` owns the active unreduced window `l..=en_u`,
  maintains `t`, `its`, `total_iter`, and `max_total = 60*n + 100`, computes
  `x/y/w`, selects `m`, scrubs sub-subdiagonal spikes, and runs the scalar
  double-shift chase.
- `eig_francis_profile_f64` already records active windows, shift samples,
  selected `m`, sweeps, deflations, exceptional shifts, fallback deflations,
  and max-total exhaustions.
- `eigvals_golden` is the strict order-sensitive bit digest gate for the
  eigenvalue stream printed by both `eigvals` and full `eig`.

## Pass 4 One-Lever Gate

The only source lever authorized by this contract is:

```text
AED-derived shift-list consumed by the existing scalar single-bulge chase.
```

The source slice must derive a bounded list of candidate double-shift packets
from a copied active Schur/AED window, validate the packet ledger, and then feed
at most one vetted packet into the existing scalar m-search and bulge-chase
schedule. The global `h` and `q_acc` update order must stay the current scalar
schedule. True multibulge chasing, far-update batching, compact-WY updates,
range/index micro-cuts, and diagnostic-only helpers are out of scope for pass 4.

Pre-score:

```text
Impact 5: measured floor is 319/1132 scalar Francis sweeps.
Confidence 3: packet validation can fence unsafe windows, but golden preservation is hard.
Effort 4: private source slice plus strict golden, focused tests, and same-worker rebench.
Score = 5 * 3 / 4 = 3.75
```

Pass 4 must reject the source hunk if any reject condition in this document
fires, or if same-worker median speed does not improve enough to keep overall
Score >= 2.0.

## Active-Window Ownership

The candidate path must use an explicit private data model equivalent to:

```text
SchurWindowShiftPlan {
    active_first: usize,       // scalar l
    active_last: usize,        // scalar en_u
    width: usize,              // active_last + 1 - active_first
    selected_m: usize,         // scalar m selected from this packet and current h
    iteration_in_window: usize,// current its before increment
    accumulated_shift: f64,    // current t after any scalar exceptional update
    x: f64,
    y: f64,
    w: f64,
    packet_index: usize,
    packet_count: usize,
}
```

Boundary rules:

- `active_first == l`, `active_last == en_u`, and `width >= 4` for any
  candidate packet. Widths that are already 1x1 or 2x2 deflation cases use the
  current scalar deflation code only.
- The copied Schur/AED window is read from
  `h[active_first..=active_last, active_first..=active_last]` before any global
  mutation. The copy must be bit-consistent with those source slots at the time
  of copy.
- Candidate planning may read the copied window and scalar metadata only. It
  may not mutate global `h`, `q_acc`, `eigenvalues`, `t`, `its`, `en`, or
  `total_iter`.
- The packet is consumed only after the existing scalar m-search recomputes a
  valid `selected_m` for the packet. A stored or predicted `selected_m` mismatch
  forces scalar fallback for that sweep.
- For `eigvals` (`want_vectors == false`), the global scalar row writes remain
  `j in k..=active_last`; slots in already-deflated trailing blocks
  `active_last + 1..n` are immutable for the active sweep.
- For full `eig` (`want_vectors == true`), scalar row writes remain `j in k..n`,
  `q_acc` remains a pure sink, and the existing ordered rotation stream is
  replayed exactly as the current scalar chase records it.
- Column writes remain the scalar rule: rows `0..=min(active_last, k + 3)` and
  columns touched by the current 2- or 3-vector reflector (`k`, `k+1`, and
  `k+2` when `notlast`).
- Eigenvalue slot writes are owned only by the scalar 1x1 and 2x2 deflation
  branches. The shift-list planner never writes `eigenvalues`.

## Hessenberg Invariants

Before accepting a candidate packet:

- Every inspected value in the copied window must be finite.
- The global active window must be upper Hessenberg: for all
  `i > j + 1` inside `active_first..=active_last`, `h[i*n + j]` must be exactly
  zero or must be a slot the current scalar cleanup would overwrite before it is
  read.
- Boundary couplings must be finite:
  `h[active_first*n + active_first]`, `h[active_last*n + active_last]`, each
  active subdiagonal `h[i*n + (i-1)]`, and the entries used by the current
  scalar shift source.
- The candidate packet must not depend on any sub-subdiagonal value.

After consuming a candidate packet through the scalar chase:

- The existing scalar cleanup remains authoritative:
  `h[i*n + (i-2)] = 0.0` for `i in (m+2)..=active_last`, and
  `h[i*n + (i-3)] = 0.0` for `i != m+2`.
- No value below the first subdiagonal in the active window may become finite
  non-zero except transiently inside the scalar chase before the existing scrub.
- Any post-sweep invariant failure rejects the candidate path and requires the
  source hunk to be removed, not patched around by tolerating dirty Hessenberg
  state.

Copied-window consistency:

- The copy must be generated from the current `h` after scalar deflation tests
  and before any candidate packet is selected.
- If scalar deflation changes `en`, `l`, or `t`, any cached copy for the old
  window is invalid.
- Cached packets are scoped to exactly one active window and one value of
  `iteration_in_window`; they must not survive a deflation or exceptional shift.

## Exceptional Shift and Max-Total Accounting

The scalar accounting remains the source of truth:

- Preserve the exceptional cadence exactly: `its == 10 || its == 20`.
- Preserve the accumulated shift `t`: when the scalar exceptional branch fires,
  it performs `t += x`, subtracts `x` from diagonals `0..=en_u`, then sets the
  exceptional `x/y/w`. Candidate packets must not bypass or reorder this.
- Preserve `its += 1` and `total_iter += 1` as one attempted scalar sweep per
  consumed packet.
- Preserve `max_total = 60*n + 100`.
- Preserve fallback deflation when `its >= 30 || total_iter >= max_total`.
- Preserve `fallback_deflations` and `max_total_exhaustions` counters in the
  profile path.

Pass 4 must force scalar behavior when:

- `iteration_in_window == 10 || iteration_in_window == 20`.
- `its >= 30` or `total_iter >= max_total` before packet consumption.
- Candidate planning would need to change `t`, change the diagonal shift
  subtraction, or change the profile counter sequence.
- Candidate planning cannot prove exactly one `record_sweep` and one
  `record_shift` equivalent for the attempted sweep.

## Strict Scalar Fallback

Fallback means calling the current scalar shift-source and scalar chase with no
changed arithmetic order, no changed read/write bounds, and no changed trace
accounting. Fallback must be bit-exact to current `eig_francis_schur_traced`.

The candidate path must force fallback for any of these conditions:

- `width < 4`.
- `active_first > active_last`, `active_last >= n`, or any boundary arithmetic
  overflows.
- `want_vectors` mode cannot be proven against the same packet path; do not
  silently use an eigvals-only proof for full `eig`.
- Any copied-window value, packet value (`x`, `y`, `w`), normalization, or
  reflector scalar is non-finite.
- Packet count is zero, exceeds the hard packet cap, or requires unbounded
  retry.
- Scalar m-search normalization would divide by zero.
- Stored `selected_m` and recomputed scalar `m` disagree.
- The packet would read or write outside the active-window boundaries above.
- The Hessenberg invariant checks fail before or after the scalar chase.
- Exceptional-shift cadence or max-total accounting is active.
- The profile sink would report different deflation totals for a pure fallback
  run.

Fallback is not a performance result. If pass 4 always falls back, the source
lever has no keep path and must be reported as rejected or zero-change.

## Ordering and Complex-Pair Slots

The eigenvalue stream order is part of the contract:

- Deflation remains bottom-up by `en`.
- A 1x1 real root writes:
  `eigenvalues[2*en] = h[en,en] + t`,
  `eigenvalues[2*en + 1] = 0.0`, then decrements `en` by one.
- A 2x2 real pair (`q >= 0.0`) writes the upper slot `na = en - 1` first with
  `r1`, then the lower slot `en` with `r2`; both imaginary slots are zero.
- A 2x2 complex pair (`q < 0.0`) writes:
  `na -> (xx + pp, +zz)` and `en -> (xx + pp, -zz)`.
- No sorting, tie-breaking repair, pair swapping, or normalization of signed
  zero/NaN payloads is allowed.
- Full `eig` may standardize the 2x2 block and queue `q_acc` rotations only
  through the existing scalar branch. Candidate planning never writes complex
  slots or Schur-vector slots directly.

## Floating-Point Ledger

Fallback ledger:

- Bit-exact to the current scalar `eig_francis_schur_traced` implementation.
- No reassociation.
- No new parallelism.
- No FMA-only assumption.
- No changed row/column bounds.
- No RNG.

Candidate ledger:

- The only permitted arithmetic-order difference is inside the copied
  Schur/AED window used to derive `x/y/w` packets. The global `h` and `q_acc`
  updates must still use the current scalar single-bulge chase.
- Any candidate packet can change convergence path. That is allowed only if the
  strict `eigvals_golden` stdout SHA, focused eig/eigvals tests, ordering
  checks, and profile accounting checks pass after the source hunk.
- No randomized AED, randomized shifts, randomized sampling, or seed-dependent
  behavior is allowed.
- If the candidate path introduces tolerance-based acceptance for local window
  deflation, that is a separate future lever and is rejected for pass 4.

## RNG Absence

The pass-4 source slice must be fully deterministic:

- No RNG, randomized AED, randomized shift sampling, or seed parameter.
- No hash-map iteration order or thread scheduling may affect packet choice.
- The packet list is a pure function of the copied active Hessenberg window,
  scalar counters (`active_first`, `active_last`, `its`, `total_iter`, `t`),
  and fixed packet caps.
- Re-running the same input at the same source revision must produce the same
  packet ledger before any benchmark result can be trusted.

## Golden SHA Gate

Pass 4 must regenerate strict golden stdout and verify:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

The gate covers the printed deterministic lines from `eigvals_golden`, including
both `eigvals_digest` and `eig_digest` for n=64, n=128, and n=256. A digest drift
is an immediate reject even if focused tests pass.

## Pass 4 Reject Conditions

Reject and remove the source hunk if any of these occur:

- Strict golden SHA drift.
- Bottom-up slot-order drift, real-pair order drift, or complex-conjugate slot
  convention drift.
- Any focused eig/eigvals test failure.
- Any finite/Hessenberg invariant failure.
- Any fallback, exceptional-shift, `total_iter`, `max_total`, or profile counter
  mismatch that cannot be explained by the candidate path and accepted by the
  golden gate.
- Any local-fallback or cross-worker benchmark used as keep proof.
- Same-worker median regression or Score < 2.0.
- Any attempt to replace the planned lever with range/index micro-cuts,
  diagnostic-only helpers, direct two-bulge chasing, or far-update batching.

## Validation Checklist for Pass 4

Behavior and proof:

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p ft-kernel-cpu --lib eig -- --nocapture`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo run --release -q -j 1 -p ft-kernel-cpu --example eigvals_golden`
- Extract deterministic golden lines and verify the SHA above.
- Run `eig_francis_profile_f64` coverage through `eig_timing_probe` or focused
  tests and confirm:
  `deflated_eigenvalue_count() == n`,
  `max_total_exhaustions <= fallback_deflations`,
  exceptional-shift packet count matches packets with `exceptional=true`, and
  each recorded selected `m` lies inside the recorded active window.

Source quality, only if production source changes:

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p ft-kernel-cpu --lib --examples --benches`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -j 1 -p ft-kernel-cpu --lib --examples --benches -- -D warnings`
- `rch exec -- cargo fmt -p ft-kernel-cpu --check`
- `ubs crates/ft-kernel-cpu/src/lib.rs`

Performance:

- Establish an immediate before row via RCH before editing or before scoring the
  final hunk.
- Re-run the after row on the same worker.
- Primary row: `eigvals_f64_256x256`.
- Supporting row: `eig_f64_256x256`.
- Keep only with same-worker evidence and Score >= 2.0. Cross-worker or
  local-fallback numbers are routing evidence only.

## Isomorphism Proof Template for Pass 4

```text
## Change: AED-derived shift-list packet consumed by scalar Francis chase

- Lever boundary:
  [single source hunk / helper name, packet cap, active-window gate]
- Ordering preserved:
  [bottom-up en deflation unchanged; 1x1/2x2 slot writes unchanged; no sorting]
- Tie-breaking unchanged:
  [no eigenvalue reordering; scalar deflation tests and m-search rules retained]
- Complex-pair convention:
  [na gets +imag, en gets -imag; real pair r1/r2 slot order unchanged]
- Floating-point:
  fallback bit-exact; candidate reassociation limited to copied-window packet
  extraction; global h/q_acc updates use existing scalar chase order
- RNG:
  none
- Fallback proof:
  [list every fallback condition exercised or unit-checked; golden fallback SHA]
- Golden outputs:
  eigvals_golden strict stdout sha256 =
  24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
- Profile accounting:
  [sweeps, defl1, defl2, fallback, exceptional, max_total_exhaustions]
- Same-worker benchmark:
  before [worker, interval]; after [worker, interval]; median speedup [x]
- Decision:
  KEEP only if Score >= 2.0 and no reject condition fired
```

## Pass 3 Decision

The proof contract clears pass 4 to attempt exactly the AED-derived shift-list
consumed by the existing scalar single-bulge chase, with pre-score 3.75. The
contract does not authorize a public dispatch change, multibulge chase,
far-update batching, range/index micro-cut, or diagnostic-only helper.
