# frankentorch-ct2yy pass 2 primitive selection

Date: 2026-06-11
Scope: planning artifact only. No source edits.

## Grounding

- Live bead: `frankentorch-ct2yy`, QR panel-factorization / recursive-panel route.
- Current code surface: `crates/ft-kernel-cpu/src/lib.rs` in `qr_householder_panel_blocked` and `qr_contiguous_f64`.
- Required alien sources:
  - `/data/projects/alien_cs_graveyard/alien_cs_graveyard.md` section 9.6: communication-avoiding algorithms, CA-QR/TSQR, Householder replay down a tree, BLAS-3 inner kernels, numerical stability/tolerance contracts.
  - `/data/projects/alien_cs_graveyard/alien_cs_graveyard.md` section 7.2: cache-oblivious recursive decomposition and cache-miss profiling discipline.
  - `/data/projects/alien_cs_graveyard/high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md`: artifact graph, no claim without evidence artifacts, contract template, opportunity score gate, profile -> prove -> one lever -> verify -> re-profile.

## Measured Evidence

Pass 1 captured routing-only Criterion evidence for the two QR rows:

| Row | Estimate |
| --- | ---: |
| `qr_f64_512x512` | `[53.778 ms 54.462 ms 55.012 ms]` |
| `qr_f64_tall_2048x128` | `[28.818 ms 29.168 ms 29.767 ms]` |

Caveat: the remote-required run was refused by RCH before execution, and the successful run used local fallback through `rch exec`. These numbers are useful for routing and scale selection only. A pass 3/4 keep or reject decision still needs same-worker remote before/after evidence.

The live tracker already records a negative result: outer `NB=32 -> 64` regressed (`qr_512` about 53 ms -> 84 ms; tall about 27 ms -> 52 ms). Current source already has compact-WY blocking for the trailing update and reverse `dorgqr`; the residual is the scalar Householder panel factorization inside `qr_householder_panel_blocked`, especially the per-reflector norm/dot/update over same-panel columns before the trailing GEMM.

## Current Implementation Facts

- Outer QR path dispatches to blocked compact-WY only for `m >= 128 && k >= 16`.
- Outer panel width is fixed at `NB = 32`.
- The shipped blocked path already:
  - forms a panel `V` and compact-WY `T`;
  - applies the trailing `R[:, pe:n]` update via `gemm::dgemm`;
  - builds reduced/full `Q` through reverse block replay at the requested output width.
- The remaining panel loop still applies each newly formed reflector to only the remaining columns of the same panel with scalar dot/update loops.

## Candidate Matrix

Score uses the optimization skill rule: `Impact * Confidence / Effort`, each factor 1..5.

| Candidate | Impact | Confidence | Effort | Score | Ranking Notes |
| --- | ---: | ---: | ---: | ---: | --- |
| A. Recursive compact-WY panel factorization (`dgeqrt`-style) inside the existing outer `NB=32` panel | 4 | 4 | 4 | 4.00 | Best pass 3/4 lever. It targets the measured residual directly while preserving the existing trailing GEMM and reverse `dorgqr` machinery. |
| B. Tall-only sequential TSQR / CAQR row tree for reduced `m >> n` QR | 5 | 3 | 5 | 3.00 | Strong alien-source fit for `2048x128`, but larger API-surface risk because Q replay, signs, and square QR behavior need separate handling. Use if A fails or tall QR becomes the sole profile target. |
| C. Cache-oblivious row-blocked panel dot/update tree with deterministic reductions | 3 | 3 | 4 | 2.25 | Attacks panel locality without changing outer blocking, but the reduction tree changes FP association and may not expose enough BLAS-3 work at `NB=32`. Research fallback, not first lever. |
| D. CholeskyQR / normal-equations QR shortcut for well-conditioned panels | 4 | 1 | 3 | 1.33 | Reject. It changes numerical behavior and rank/conditioning semantics; it is not an isomorphic Householder QR route. |

## Selected One-Lever Candidate

Select candidate A for pass 3/4: a recursive compact-WY panel factorization inside `qr_householder_panel_blocked`.

One lever only:

1. Keep outer `NB = 32`, the existing `m >= 128 && k >= 16` dispatch, the existing trailing update, and the existing reverse `dorgqr`.
2. Replace only the intra-panel factorization strategy with a recursive or leaf-blocked helper that:
   - factors a small leaf block of panel columns with the current Householder sign/tiny policy;
   - forms the leaf compact-WY `T`;
   - applies that leaf block to the remaining columns of the same outer panel in a blocked operation;
   - emits aggregate `V/T` compatible with the existing trailing update and Q replay.

Why this is not NB retuning:

- The outer panel width remains `32`.
- No dispatch threshold changes are part of the lever.
- The rejected `NB=64` family tried to make the existing panel/trailing split wider; this candidate changes the algorithm used to factor the current panel itself.
- The expected win comes from converting same-panel scalar Householder application into recursive/block reflector work, not from making the GEMM trailing update larger.

Fallback trigger:

- If any QR golden/unit check fails, reject the source hunk and keep this artifact as routing evidence.
- If `A = Q*R`, `Q^T*Q = I`, upper-triangular cleanup, shape/order, or near-zero-column behavior drift beyond the current tolerance policy, reject.
- If same-worker remote Criterion cannot be obtained, do not keep or reject; record as routing-only.
- If same-worker remote A/B does not show at least a clear `qr_f64_512x512` median win with no material `qr_f64_tall_2048x128` regression, reject and route to candidate B.

## Isomorphism Obligations

Householder sign policy:

- Preserve `sign = if v0 >= 0.0 { 1.0 } else { -1.0 }`.
- Preserve `diag = -sign * norm` and below-diagonal zeroing after each accepted reflector.
- Preserve the current `tiny = f64::EPSILON * 1e6` skip behavior.
- Do not introduce pivoting or sign reorientation unless a deterministic post-normalization proves equivalence to the current public convention.

Q/R shape and order:

- `reduced=true`: `Q` remains row-major `m x k`; `R` remains row-major `k x n`.
- `reduced=false`: `Q` remains row-major `m x m`; `R` remains row-major `m x n`.
- Column order, reflector order, and row-major output order must remain stable.
- Empty input and one-dimensional rejection behavior are unchanged.

Rank behavior:

- QR remains unpivoted Householder QR.
- Near-zero columns keep the current tau-zero/skip behavior.
- No rank-revealing QR, no column pivoting, no CholeskyQR fast path, no SVD fallback, and no condition-based API behavior changes.

Floating-point policy:

- Current blocked QR is already tolerance-ledger work, not bit-exact relative to the old scalar sweep.
- The candidate must preserve deterministic operation order for a given input and thread topology.
- Required tolerance checks:
  - deterministic fixture comparison against the current implementation for representative square and tall matrices;
  - `A = Q*R` within the existing QR tolerance envelope (`1e-6` scale for tall blocked fixtures, tighter where current tests require it);
  - `Q^T*Q = I` within the existing orthonormality envelope (`1e-8` to `1e-9`);
  - strict zero below diagonal after cleanup.

RNG absence:

- No runtime RNG is allowed.
- Deterministic test fixture generation may use the existing fixed arithmetic patterns or fixed-seed local generators only.

Golden/check strategy for pass 3/4:

- Focused source-only diff review for `crates/ft-kernel-cpu/src/lib.rs`.
- Focused tests:
  - `cargo test -p ft-kernel-cpu qr_ -- --nocapture`
  - any existing blocked/tall QR test names if filtered command syntax needs adjustment.
- Golden probe before/after against fixed square and tall matrices:
  - reconstruction samples;
  - orthonormality samples;
  - R upper-triangular zeros;
  - output shape metadata.
- Criterion rows, same-worker remote only:
  - `qr_f64_512x512`
  - `qr_f64_tall_2048x128`
- Do not run full-workspace builds for this lane unless explicitly requested later.

## Explicit Reject List

Do not retry these adjacent levers for `ct2yy`:

- Outer `NB` retuning, including `32 -> 64`, adaptive panel-size sweeps, or dispatch-threshold-only changes.
- More trailing GEMM packing/transposition reshuffles before the panel factorization changes; the existing residual is not the trailing update.
- Rayon fan-out inside the scalar panel loop as the primary lever; dependencies are sequential across panel columns and small-panel overhead is likely to dominate.
- T-only or allocation-only cleanup (`vt`, `tt`, scratch reuse) as the next no-gaps lever; it can be considered only after a real panel algorithm lands and re-profiles.
- Any Householder sign/tiny-threshold tweak.
- CholeskyQR, Gram-Schmidt, normal-equations QR, rank-revealing QR, or column pivoting under this public `qr_contiguous_f64` path.
- Full TSQR replacement for both square and tall QR in the next pass; keep it as fallback candidate B, not the immediate one-lever implementation.
