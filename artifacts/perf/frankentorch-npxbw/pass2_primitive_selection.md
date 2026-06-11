# frankentorch-npxbw Pass 2 - Alien Primitive Selection

Date: 2026-06-11
Agent: IvoryDeer
Scope: primitive selection and proof contract only. No production source edits, no commits, no pushes.

## Measured Hotspot Evidence

Authoritative pass-1 proof is commit `03a7293f`, same-worker remote `vmi1152480`:

| Row | Worker | Interval | Median-ish |
| --- | --- | --- | ---: |
| `eigvals_f64_256x256` | `vmi1152480` | `[27.044 ms 28.425 ms 29.811 ms]` | `28.425 ms` |
| `eig_f64_256x256` | `vmi1152480` | `[56.971 ms 59.195 ms 61.497 ms]` | `59.195 ms` |

Strict fallback oracle:

- Golden artifact: `artifacts/perf/frankentorch-npxbw/pass1_eigvals_golden.strict.stdout`
- SHA-256: `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`
- Verification command: `sha256sum -c artifacts/perf/frankentorch-npxbw/pass1_eigvals_golden.strict.stdout.sha256`

Supporting routing evidence from the extra `vmi1227854` pass-1 profile artifact:

| n | `eigvals` | `eig` | `eig - eigvals` |
| ---: | ---: | ---: | ---: |
| 128 | 4.47 ms | 7.19 ms | 2.72 ms |
| 256 | 28.81 ms | 46.06 ms | 17.26 ms |
| 512 | 374.33 ms | 512.41 ms | 138.08 ms |
| 1024 | 2828.59 ms | 4784.97 ms | 1956.38 ms |

Interpretation: `eigvals` carries the shared non-symmetric Hessenberg/Francis QR floor. Full `eig` adds eigenvector machinery, but the shared values path is still the first target because a sweep-count or far-update batching win should move both rows.

## Current Code Surface

- `eig_impl` in `crates/ft-kernel-cpu/src/lib.rs` copies the input, reduces to Hessenberg form with `hessenberg_reduce_blocked` for `n >= 128`, then calls `eig_francis_schur`.
- `eig_francis_schur` is a classic EISPACK/hqr2-style implicit double-shift Francis QR loop with bottom-up deflation, exceptional shifts at iterations 10 and 20, and a scalar 3-row bulge chase.
- `want_vectors=false` already restricts the row update to `[k, en]`; `want_vectors=true` records `EigQaccOp` rotations and replays them after the full Schur chase. The shared residual is therefore inside active-window shift selection and bulge chasing, not q_acc replay alone.

## Mapped Graveyard / Artifact Sources

- `/data/projects/alien_cs_graveyard/alien_cs_graveyard.md` section 9.6, communication-avoiding algorithms: use blocked/tree/batched dense linear algebra to reduce data movement and move work from scalar BLAS-1/2 style updates toward cache-local BLAS-3 kernels.
- `/data/projects/alien_cs_graveyard/high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md`: profile-first optimization, one-lever gates, explicit fallback, proof/perf contracts, and graceful degradation are required for alien uplift.
- `/home/ubuntu/.codex/skills/alien-artifact-coding/references/34-NUMERICAL-LINEAR-ALGEBRA.md`: method selection is QR/eigendecomposition; proof obligations must include decomposition accuracy, orthogonality/Schur-form checks where applicable, condition/stability notes, convergence traces, and independent golden cross-checks.
- No external C/Fortran BLAS/LAPACK/MKL/XLA is allowed. The lever must stay in safe Rust and may only reuse the repo's existing safe Rust kernel boundaries.

## Candidate Cards

### 1. Chosen for Pass 3: deterministic shift-source + sweep-profile scaffold

Change: Extract the current double-shift source and active-window/sweep accounting into a private diagnostic/scaffold path around `eig_francis_schur`, without changing public dispatch or production results. Record shift pairs, exceptional-shift events, active-window widths, sweep counts, deflations, and fallback triggers.

Hotspot evidence: pass 1 shows `eigvals_f64_256x256` at `28.425 ms` median on `vmi1152480`, and the timing probe shows the values path grows superlinearly from n=256 to n=1024.

Mapped primitive: communication-avoiding dense linear algebra precondition for direct small-bulge multishift QR: first make the shift stream and active-window work explicit so the later multi-bulge/far-update lever has a deterministic source and a convergence trace.

Score: Impact 3 x Confidence 5 / Effort 2 = 7.50.

Why selected now: this is the safest one-lever bridge from the existing double-shift loop to multishift QR. It attacks the correct surface, does not replay rejected AED work, and gives pass 4 the exact shift/convergence artifact needed to avoid a blind rewrite.

Fallback trigger: if the scaffold changes `eigvals_golden` SHA-256, touches public dispatch, changes eigenvalue ordering, or cannot isolate the current shift sequence without altering arithmetic, reject the source hunk and keep only the planning artifact.

### 2. Deferred: 4-shift / two-bulge direct small-bulge chase

Change: Use two current-compatible double-shift pairs as a four-shift packet, introduce two small bulges, chase them through the active Hessenberg window, and batch the far row/column updates where the bulges are separated.

Hotspot evidence: targets the same `eigvals` floor. This is the first real algorithmic replacement for the scalar one-bulge loop.

Mapped primitive: LAPACK-style small-bulge multishift QR plus graveyard section 9.6 communication avoidance: keep near-bulge updates scalar and aggregate far updates into cache-local dense kernels.

Score: Impact 5 x Confidence 3 / Effort 4 = 3.75.

Why not pass 3: proof risk is too high before the shift stream and convergence counters are explicit. It can change deflation timing and eigenvalue slot order unless the pass-3 scaffold pins the contract.

Fallback trigger: reject if strict golden digest changes, if conjugate-pair slot assignment diverges, if same-worker `eigvals_f64_256x256` median speedup is below the Score>=2.0 gate, or if iteration/fallback counts exceed the current double-shift loop on the golden fixture.

### 3. Deferred/research: compact-WY accumulated reflector far updates

Change: After creating a bounded packet of small-bulge reflectors, represent the far-window transformations as compact reflectors and apply them with the existing safe Rust `gemm::dgemm` boundaries.

Hotspot evidence: same Francis QR floor, but this only pays once direct multishift packets exist and the active window is large enough to amortize setup.

Mapped primitive: communication-avoiding QR/CAQR idea from section 9.6 applied inside the Hessenberg QR stage, not to a full dense QR factorization.

Score: Impact 4 x Confidence 3 / Effort 5 = 2.40.

Why not pass 3: this composes two levers: multibulge scheduling and compact reflector batching. It also reassociates floating-point far updates, so it needs a stronger Schur-form proof and cannot be the first npxbw edit.

Fallback trigger: reject if batching changes strict digests, increases allocations enough to erase the QR win, or fails to improve both `eigvals` and full `eig` on a same-worker A/B.

## Explicit Rejections

- Rejected qglh3 families are out of scope: values-only AED suffix, whole-window threshold AED with q_acc, active-window threshold trims, and eigenvector-only machinery. They either already failed same-worker gates or do not attack the shared values floor.
- Generic TSQR/CAQR full-matrix QR is not selected for this bead. The matrix is already Hessenberg and the hotspot is implicit Schur QR, so the useful graveyard mapping is blocked/batched bulge updates, not replacing Hessenberg reduction or symmetric `eigvalsh/eigh`.
- `frankentorch-x53r3` symmetric reduction work is excluded. This pass only targets general non-symmetric `geev/eigvals`.

## Chosen Pass-3 Target

Implement at most one private deterministic shift-source and sweep-profile scaffold for `eig_francis_schur`.

Required outputs for pass 3:

- No public dispatch change unless it is a test/example-only diagnostic route.
- A pass-3 artifact recording active-window widths, shift pairs, exceptional shifts, total sweeps, deflations, and whether the scaffold exactly follows current double-shift arithmetic.
- Strict fallback proof: `sha256sum -c artifacts/perf/frankentorch-npxbw/pass1_eigvals_golden.strict.stdout.sha256` remains valid after regenerating `eigvals_golden` output.
- Focused correctness checks: `eigvals_matches_eig`, `eigvals_companion_complex_roots`, and `eig_parallel_schur_vector_update_matches_single_thread_bit_exact`.
- Crate-scoped gates only: `rch exec -- cargo check -p ft-kernel-cpu --lib --examples --benches`, focused tests, UBS on touched files, and fmt check for touched Rust files.

## Behavior / Isomorphism Proof Plan

Ordering: preserve current bottom-up `en` deflation order. Any future multishift path must write 1x1 and 2x2 roots into the same eigenvalue slots the current loop would use for the corresponding active window, or fall back to legacy double-shift.

Conjugate pairs: preserve current real-Schur convention: for a complex pair, the upper slot (`na`) carries positive imaginary part and the lower slot (`en`) carries negative imaginary part.

Tie-breaking and deflation thresholds: keep the existing `eps * s` subdiagonal split test, exceptional-shift cadence, and `max_total = 60 * n + 100` guard in strict mode. Any new adaptive cutoff must be disabled until it has its own proof bead.

Floating-point policy: pass 3 must not change arithmetic. Later multishift/far-update work may only ship if strict golden digests and focused eig tests pass; otherwise it remains behind a rejected artifact. No mixed precision and no external BLAS.

RNG: none. Shift selection is deterministic from the current active Hessenberg window.

Golden SHA: the order-sensitive strict fallback SHA stays `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.

Numerical ledger for later source levers: report convergence iterations, fallback count, residual Schur-form sanity (`H[i,j]` below subdiagonal cleaned/bounded), and agreement of `eigvals` vs `eig.eigenvalues`.

## Fallback / Reject Triggers

- Any change to the strict golden SHA-256.
- Any public behavior change to `eig_contiguous_f64` or `eigvals_contiguous_f64` during pass 3.
- Any attempt to revive the rejected qglh3 AED threshold families or the symmetric `eigvalsh/eigh` surface.
- RCH local fallback for benchmark proof. Local fallback can route, but cannot keep a source lever.
- Same-worker speedup below the Score>=2.0 gate after a future implementation pass.
- Non-convergence, increased fallback count, changed complex-pair ordering, or failure of focused eig/eigvals tests.
