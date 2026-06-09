# frankentorch-34hx9 pass 2: alien primitive selection

## Scope

- Bead: `frankentorch-34hx9`
- Target family: blocked symmetric tridiagonalization for `eigh` / `eigvalsh`
- Source edits in this pass: none
- Primary evidence: `artifacts/perf/frankentorch-34hx9/pass1_baseline_profile.md`
- Canonical primitive sources consulted:
  - `/data/projects/alien_cs_graveyard/alien_cs_graveyard.md`
  - `/data/projects/alien_cs_graveyard/high_level_summary_of_frankensuite_planned_and_implemented_features_and_concepts.md`
  - `/data/projects/.scratch/no_gaps_directive.txt`
  - `/home/ubuntu/.codex/skills/alien-artifact-coding/references/34-NUMERICAL-LINEAR-ALGEBRA.md`

## Baseline facts

All Pass 1 rows were from RCH worker `ovh-a`:

| Row | Median |
| --- | ---: |
| `eigh_f64_256x256` | 9.0613 ms |
| `eigvalsh_f64_256x256` | 5.7728 ms |
| `sym_rank2k_lower_scalar_f64_256x32` | 1.2644 ms |
| `sym_rank2k_lower_gemm_f64_256x32` | 280.49 us |

The lower rank-2k primitive is 4.51x faster via GEMM on the same worker. That is the concrete BLAS-3 building block for a blocked `dsytrd` panel/trailing-update pass.

Local code shape:

- `symmetric_rank2k_lower_update_f64` already implements `A := A - (V W^T + W V^T)` over lower storage using two GEMM-backed temporaries.
- `eigh_contiguous_f64` uses packed-lower `tred2`, backtransforms eigenvectors, runs transposed `tql2`, then sorts eigenpairs by `total_cmp`.
- `eigvalsh_contiguous_f64` uses the separate packed-lower values-only `tred2` and values-only `tql2`, then sorts values by `total_cmp`.
- Prior project notes say the packed-lower `eigvalsh` lane and packed-full `eigh` lane should be treated as distinct kernels, not one generic eigensolver surface.

## Graveyard and math mapping

The no-gaps directive points linalg work toward safe-Rust BLAS/LAPACK-class kernels: cache-blocked GEMM, blocked LU/QR/Cholesky/SVD, and strict behavior proof before shipment. The graveyard numeric-kernel playbook maps this bottleneck to cache locality, SIMD/tiled kernels, communication-avoiding dense linear algebra, and explicit numerical-stability certificates. The artifact-coding numerical-linear-algebra guide adds the required verification shape: factorization/reconstruction residuals, orthogonality checks, condition/spectral-gap notes, and fallback for unstable or ill-conditioned cases.

For this bead, the directly harvested primitive is:

`blocked Householder tridiagonalization panel + compact/WY-style trailing update`

Implementation-level artifact for the next pass should be a narrow reducer contract, not a broad eigensolver rewrite:

1. Factor a panel of Householder reflectors in the same lower-storage convention.
2. Build panel matrices `V` and `W` for the trailing submatrix.
3. Apply only the trailing lower update through the measured GEMM-backed rank-2k primitive.
4. Keep the current scalar `tred2` path as strict fallback and proof oracle.

## Candidate matrix

Scoring:

- `ESO Score = Impact * Confidence / Effort`
- `EV = Impact * Confidence * Reuse / (Effort * AdoptionFriction)`
- `Impact`, `Confidence`, `Reuse`, `Effort`, and `AdoptionFriction` are scored 1-5. Higher effort/friction is worse.
- Keep gate for implementation: `ESO Score >= 2.0`, plus a public-row same-worker benchmark win after implementation.

| Candidate | Impact | Conf. | Reuse | Effort | Friction | ESO Score | EV | One-lever now? | Expected affected rows | Proof burden | Rejection triggers |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| (a) Full blocked `dsytrd` panel integration for full `eigh` | 5.0 | 3.0 | 5.0 | 5.0 | 4.0 | 3.00 | 3.75 | Not as the next pass. Correct primitive, but full vectors add reflector persistence, backtransform, sign/orientation, and tolerance policy in one large surface. | Primary `eigh_f64_256x256`; secondary `eigvalsh_f64_256x256` if shared reducer is touched; guard `sym_rank2k_lower_*`. | Highest: FP reassociation, eigenvector basis/sign changes, degenerate eigenspaces, reconstruction/orthogonality oracle, strict fallback. | Score < 2.0 after same-worker A/B; any reconstruction or orthogonality regression; eigenvalue order/tie divergence beyond policy; fast path cannot be isolated behind fallback. |
| (b) Values-only blocked trailing-update lane for `eigvalsh` | 4.0 | 4.0 | 4.0 | 3.0 | 2.0 | 5.33 | 10.67 | Yes. This is the narrowest public-row integration of the proven rank-2k primitive and avoids eigenvector orientation/backtransform until the reducer is proven. | Primary `eigvalsh_f64_256x256`; guard `eigh_f64_256x256` for accidental drift; guard `sym_rank2k_lower_*` for primitive regression. | Medium: eigenvalue ordering and FP tolerance ledger; no eigenvector sign/orientation burden because vectors are not returned; strict fallback keeps bit-exact path. | `eigvalsh_f64_256x256` does not reach Score >= 2.0 on same-worker A/B; values differ from strict/oracle beyond tolerance; small/ill-conditioned cases need fallback too often; rank-2k guard regresses materially. |
| (c) Two-stage band reduction | 5.0 | 2.0 | 5.0 | 5.0 | 5.0 | 2.00 | 2.00 | No for the next one-lever pass. It is an alien-grade deeper primitive, but it needs band reduction plus second-stage bulge chasing and new proof fixtures. | Potentially both `eigh_f64_256x256` and `eigvalsh_f64_256x256`; no direct current primitive row. | Very high: two reductions, bulge-chase correctness, FP/tolerance policy, eigenvectors if full path, fallback. | First-stage band row does not beat blocked one-stage dsytrd; second-stage dominates; proof cannot isolate one change; constants lose at n=256. |
| (d) Tridiagonal divide-and-conquer / secular solve | 5.0 | 2.5 | 4.0 | 5.0 | 4.0 | 2.50 | 2.50 | No for this pass. It attacks the solver/backtransform residual, but Pass 1 evidence selected tridiagonalization and rank-2k first. Needs a fresh QL-vs-reduction split before implementation. | Mainly `eigh_f64_256x256`; possible `eigvalsh_f64_256x256` if values-only secular solver is included. | Very high: secular equation convergence, deflation tie policy, eigenvector merge orientation, tolerance oracle, degenerate spectra. | Fresh profile shows QL is not dominant; deflation changes ordering/ties; eigenvector reconstruction fails; implementation cannot be split from blocked reduction. |
| (e) Improve only `symmetric_rank2k_lower_update_f64` temp allocation/accumulation | 1.5 | 3.5 | 3.0 | 2.0 | 3.0 | 2.63 | 2.63 | No as a standalone pass. It is primitive-local and does not touch public `eigh`/`eigvalsh` until blocked reduction is integrated. | `sym_rank2k_lower_gemm_f64_256x32` only; no public-row change unless later integrated. | Low-medium: rank-2k numerical tolerance vs scalar reference, allocation behavior, no eigensolver semantics. | Primitive row improves but public rows do not move; temp changes obscure later blocked-reducer proof; numerical drift grows relative to existing helper. |

## Selected candidate for Pass 3/4

Selected: **(b) values-only blocked trailing-update lane for `eigvalsh`**.

This is the highest-scoring one-lever wedge that is both profile-backed and immediately implementable without editing the full eigenvector surface. It should be implemented as a blocked values-only `dsytrd` reducer behind the current scalar values-only reducer, using the measured GEMM-backed lower rank-2k helper for trailing updates. It is not the endpoint: it is the first proof-carrying integration of the blocked primitive that can later be promoted to the full `eigh` vector path.

Recommendation contract:

- Workload: symmetric well-conditioned `eigvalsh_f64_256x256` from `linalg_bench`.
- Baseline: 5.7728 ms median on `ovh-a`.
- Primitive guardrail: `sym_rank2k_lower_gemm_f64_256x32` 280.49 us vs scalar 1.2644 ms on `ovh-a`.
- One lever: add a blocked values-only tridiagonalization path with GEMM-backed lower trailing update; do not change QL, sorting, full-vector backtransform, or rank-2k helper internals in the same commit.
- Fallback: current `eigh_tred2_values_only` remains the strict/reference path and is used for small sizes, non-finite inputs, failed proof gates, or explicit strict mode.
- Acceptance: same-worker RCH A/B on `eigvalsh_f64_256x256` yields Score >= 2.0; `eigh_f64_256x256` is unchanged within noise unless the implementation deliberately shares code; all proof fixtures pass.
- Rejection: remove the source hunk if public-row win is noise-scale, if fallback is required for the benchmark matrix, or if value ordering/tolerance proof fails.

## Proof implications for the selected pass

Ordering and tie-breaking:

- Preserve ascending `total_cmp` output order.
- For exact equal values, the values-only output has no eigenvector association, so ordering is observable only as the sorted value sequence.
- Any full-path promotion later must preserve eigenpair sorting by `(eigenvalue total_cmp, original index/tie ledger)` or explicitly document a tolerance-equivalent tie policy.

Floating point:

- Blocked `dsytrd` reassociates dot products and trailing updates through GEMM, so bit-exact equality with the scalar EISPACK path is not expected.
- The fast path must have an explicit tolerance ledger: max absolute error, max relative error, reconstruction residual for the full path later, and condition/spectral-gap notes for stress cases.
- Strict fallback keeps the current scalar path for bit-exact mode and for proof failures.

Eigenvector sign/orientation:

- The selected values-only pass returns no eigenvectors, so it avoids sign/orientation obligations now.
- A later full `eigh` promotion must compare eigenvectors sign-insensitively for non-degenerate spectra, validate `V^T V ~= I`, validate `A ~= V diag(lambda) V^T`, and handle degenerate eigenspaces by subspace projection rather than column identity.

RNG:

- No RNG is involved in the current benchmark matrix or selected algorithm. The pass must not introduce randomized pivoting, randomized sketches, or nondeterministic work scheduling.

Golden SHA and oracle strategy:

- Before any source edit, generate strict-path goldens for representative symmetric matrices at n=32/64/128/256 and record sha256.
- After the source edit, verify strict fallback goldens are unchanged byte-for-byte.
- For the fast blocked path, record a separate tolerance oracle against the strict scalar reducer and, where available, NumPy/PyTorch `eigvalsh` for the same matrices.
- Required focused tests: `eigvalsh_matches_eigh`, `eigh_tred2_tql2_orthonormal_and_reconstructs_24x24` as a guard, the rank-2k scalar-reference test, and a new blocked-values-only oracle fixture.

## Pass 3/4 execution sketch

1. Re-run the Pass 1 public pair and rank-2k guardrail on one RCH worker before editing.
2. Add only the blocked values-only reducer and strict fallback gate.
3. Generate and check strict-path sha256 goldens, plus tolerance oracle output for the fast path.
4. Run crate-scoped validation only: `ft-kernel-cpu` focused tests, `cargo check -p ft-kernel-cpu --all-targets`, `cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`, and `cargo fmt --check`.
5. Re-benchmark on the same worker; keep only if `eigvalsh_f64_256x256` clears Score >= 2.0 and no proof gate weakens.
