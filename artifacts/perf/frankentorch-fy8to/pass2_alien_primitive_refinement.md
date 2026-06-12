# frankentorch-fy8to pass 2 alien primitive refinement

Date: 2026-06-12T22:21:40Z

Scope: artifact/proof-contract/progress only for bead `frankentorch-fy8to`.
No production source was edited in this pass.

## Inputs

Baseline/profile evidence comes from `pass1_baseline_profile.md`:

| Worker | Row | Criterion interval |
|--------|-----|--------------------|
| `vmi1149989` | `eigvals_f64_256x256` | `[33.768 ms 35.625 ms 37.476 ms]` |
| `vmi1149989` | `eig_f64_256x256` | `[50.173 ms 51.101 ms 52.098 ms]` |
| `vmi1227854` supplemental | `eigvals_f64_256x256` | `[24.692 ms 25.049 ms 25.415 ms]` |
| `vmi1227854` supplemental | `eig_f64_256x256` | `[73.003 ms 75.979 ms 79.258 ms]` |

Strict golden SHA:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725
```

Profile evidence:

| n | sweeps | defl1 | defl2 | fallback | exceptional |
|---|--------|-------|-------|----------|-------------|
| 256 | 319 | 14 | 121 | 0 | 0 |
| 1024 | 1132 | 18 | 503 | 0 | 0 |

Source anchors in `crates/ft-kernel-cpu/src/lib.rs`:

- `eig_impl` reduces to Hessenberg, then calls `eig_francis_schur`.
- `eig_francis_schur_traced` owns the active unreduced window through `l..=en_u`.
- The current scalar path records active windows, shift packets, selected `m`, sweeps, 1x1/2x2 deflations, exceptional shifts, and max-total fallbacks.
- The current row/column modification order is scalar EISPACK-style double-shift Francis QR; q_acc is a sink and is replayed later only when `want_vectors`.

Canonical graveyard mapping:

- `alien_cs_graveyard.md` section 9.6, Communication-Avoiding Algorithms: use panel/window ownership and dense inner kernels to reduce data movement, but carry explicit numerical stability and convergence certificates.
- FrankenSuite summary section 0 methodology: profile-first optimization, opportunity-matrix gates, proof/perf/decision/observability contracts, and graceful degradation.
- Alien-artifact numerical-linear-algebra family: matrix characterization, convergence traces, factorization/eigenvalue accuracy checks, conditioning/tolerance ledgers, and failure-mode fallbacks.

## Recommendation card

Change:

Build a strict-fallback Schur-window / AED-derived shift-list primitive for the non-symmetric Francis QR core. The first source slice should not be another direct two-bulge rewrite. It should introduce a bounded active-window plan that derives a short list of vetted double-shift packets from a copied Schur/AED window, then feeds those packets through the existing scalar single-bulge chase order. Unsupported windows fall back to the exact current shift source and scalar loop.

Hotspot evidence:

- `eigvals_f64_256x256` remains a profile-backed target with same-worker baseline medians of 35.625 ms on `vmi1149989` and supplemental 25.049 ms on `vmi1227854`.
- `eig_timing_probe` shows 319 sweeps at n=256 and 1132 sweeps at n=1024, with no fallback and no exceptional shifts. The win must reduce convergence work or active-window traversal, not shave an index range.
- The pass-6 predecessor rejected direct two-bulge/four-shift source editing because the current scalar loop interleaves row and column updates plus deflation after each single-bulge sweep.

Mapped graveyard sections:

- Communication-avoiding algorithms: treat the active Hessenberg window as the communication boundary, harvest a shift list from a compact local Schur/AED window, and only then spend global sweeps.
- FrankenSuite methodology: one lever, same-worker benchmark, explicit fallback trigger, opportunity score, and proof artifacts.
- Numerical linear algebra proof obligations: Hessenberg invariant, convergence accounting, eigenvalue slot ordering, complex-pair representation, tolerance/floating-point ledger, and golden-output verification.

EV / ESO score:

- ESO score for the selected next source lever: `Impact 5 * Confidence 3 / Effort 4 = 3.75`.
- Graveyard EV framing: high impact because sweep count is the measured floor; moderate confidence because strict golden preservation is difficult; high effort because active-window ownership and fallback accounting must be explicit.

Priority tier: A. It is the next profile-backed route after range micro-cuts and direct two-bulge edits failed the keep gate.

Adoption wedge:

Private `ft-kernel-cpu` subprimitive behind strict runtime gates. The public `eig_impl`, `eig_contiguous_f64`, and `eigvals_contiguous_f64` surfaces stay unchanged until the golden and same-worker gates pass.

Budgeted mode:

- Window size cap: initially active windows with width >= 128 and no exceptional shift pressure.
- Shift-list cap: bounded packet count per active window; no unbounded retries.
- On exhaustion, malformed candidate, non-finite packet, unsupported complex pairing, Hessenberg violation, or max-total pressure: use the existing scalar shift source for that window.

Expected-loss model:

States:

- `S0`: scalar fallback is required.
- `S1`: shift-list candidate is structurally valid but unproven.
- `S2`: shift-list candidate is structurally valid and strict-golden clean.

Actions:

- `A0`: scalar fallback.
- `A1`: consume AED-derived shift list through existing scalar single-bulge chase.
- `A2`: later, after separate proof, consider true multibulge/far-update scheduling.

Loss:

- Any golden SHA drift, ordering drift, fallback-accounting drift, or unsupported window treated as dominant loss and forces `A0`.
- Runtime non-win on same-worker benchmark forces rejection even if behavior is preserved.

Calibration and fallback trigger:

Pass 3 must compile a witness ledger for every candidate packet. Pass 4 may only use `A1` when the ledger proves active-window boundaries, finite packet values, preserved deflation accounting, and strict fallback eligibility.

Isomorphism proof plan:

- Fallback path must be bit-exact to the current scalar source.
- Candidate path must preserve output slot order and complex-conjugate pair slot convention.
- Golden output must hash to `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
- Focused eig/eigvals tests and the profile helper's deflated eigenvalue count must agree with `n`.

Before/after target:

- Primary row: `eigvals_f64_256x256`.
- Before: `vmi1149989` median 35.625 ms; supplemental `vmi1227854` median 25.049 ms.
- Keep target: same-worker median speedup at least 1.15x and overall Score >= 2.0 with no golden drift.

Primary failure risk and countermeasure:

Risk: a shift list changes the convergence path, which can perturb floating-point results, deflation order, or complex-pair slot assignment.

Countermeasure: strict scalar fallback for any unsupported window, plus a pass-3 ledger that records active-window ownership, packet origin, deflation accounting, and golden SHA gate before pass-4 source use.

Baseline comparator:

Current `eig_francis_schur_traced` scalar EISPACK-style double-shift loop.

Rollback:

Revert the pass-4 source hunk only; pass-2/pass-3 artifacts remain as negative or routing evidence.

## Opportunity matrix

| Candidate | Impact | Confidence | Effort | Score | Decision |
|-----------|--------|------------|--------|-------|----------|
| AED-derived shift-list consumed by the existing scalar single-bulge chase | 5 | 3 | 4 | 3.75 | Select for pass 4 only if pass 3 proof contract clears it. |
| Standalone Schur-window plan/witness builder with strict scalar fallback gates | 4 | 4 | 4 | 4.00 | Required pass-3 contract artifact; enables the selected source lever but is not a standalone performance keep. |
| True Schur-window multibulge/far-update kernel after the shift-list proof | 5 | 2 | 5 | 2.00 | Later candidate after shift-list correctness; not pass 4. |
| Range/index micro-cuts inside the existing row/column loops | 1 | 1 | 1 | 1.00 | Reject. Prior range cuts regressed and do not attack the sweep-count floor. |
| Diagnostic-only shift helpers with no runtime consumption path | 1 | 5 | 2 | 2.50 nominal, but no keep path | Reject for this bead. Diagnostics already exist; this pass needs a source-ready proof contract. |
| Direct two-bulge/four-shift source edit in the current scalar loop | 5 | 1 | 4 | 1.25 | Reject. It cannot preserve strict fallback because the current loop interleaves single-bulge row/column updates and deflation decisions after each sweep. |
| Eigenvector/q_acc-only work | 1 | 4 | 2 | 2.00 nominal, wrong row | Reject. `eigvals` is the target, and q_acc is skipped for `want_vectors=false`. |

## Pass 3 proof contract

Pass 3 should write the proof contract and, if source inspection requires it, only a non-production witness specification. It must not advance to a source optimization.

Required proof fields:

- Active-window boundary ownership: every candidate declares `active_first`, `active_last`, selected `m`, and which matrix slots may be read or written. Already-deflated trailing blocks remain immutable for `eigvals`.
- Hessenberg invariant checks: before and after any candidate window, all entries strictly below the first subdiagonal must be zero or explicitly scrubbed exactly as the scalar path does; candidate packets must not rely on dirty sub-subdiagonal state.
- Exceptional-shift and max-total accounting: preserve the current `its == 10 || its == 20` exceptional cadence, `total_iter`, `max_total = 60*n + 100`, fallback deflation counts, and max-total exhaustion counts.
- Strict scalar fallback: unsupported width, non-finite shift packet, zero normalization, failed Hessenberg invariant, selected-m mismatch, exceptional-shift pressure, or budget exhaustion must route through the current scalar shift and chase order.
- Ordering and complex-pair slots: preserve bottom-up `en` deflation order, 1x1 real slot writes, 2x2 real-pair formulas, and complex conjugate pair slots `(real,+imag)` then `(real,-imag)`.
- Floating-point ledger: fallback performs no reassociation, no FMA-only assumption, no changed reduction order, and no new parallelism. Candidate path must document each intentional arithmetic-order difference.
- RNG absence: no randomized AED, no randomized shifts, no sampling-dependent behavior.
- Golden SHA gate: strict stdout hash must remain `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
- Focused tests: `cargo test -p ft-kernel-cpu --lib eig -- --nocapture`, strict `eigvals_golden`, and `eig_francis_profile_f64` deflated-count/accounting checks.
- Quality gates for any later source change: crate-scoped `rch exec -- cargo check -p ft-kernel-cpu --lib --examples --benches`, clippy on the same surface, `cargo fmt -p ft-kernel-cpu --check`, and UBS on changed files.
- Benchmark gate: before/after on the same RCH worker; no cross-worker keep decisions.

## Pass 4 pre-score

Lever to attempt if pass 3 clears the contract:

`AED-derived shift-list consumed by the existing scalar single-bulge chase`.

Plan:

1. For eligible active windows, derive a bounded list of double-shift packets from a copied trailing Schur/AED window.
2. Validate the packet ledger against active-window, Hessenberg, finite-value, and fallback rules.
3. Consume at most one vetted packet per scalar sweep through the existing row/column chase schedule.
4. Fallback to the current scalar shift source whenever the ledger is incomplete or invalid.

Pre-score:

```text
Impact 5: sweep count is the measured floor at n=256/n=1024.
Confidence 3: the proof contract can fence unsafe windows, but golden preservation is hard.
Effort 4: private source slice plus proof/golden/same-worker benchmark.
Score = 5 * 3 / 4 = 3.75
```

Reject condition:

Any strict golden drift, slot-order drift, fallback-accounting drift, focused eig failure, or same-worker median regression rejects the source hunk. Do not replace it with another range/index micro-cut.
