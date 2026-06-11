# frankentorch-qglh3 Pass 2 AED Contract

Bead: `frankentorch-qglh3`
Date: 2026-06-11

## Measured Failure Signature

The current profile-backed residual is the non-symmetric Francis QR floor:

- `eigvals_f64_256x256` on `vmi1227854`: `[26.514 ms 27.445 ms 29.029 ms]`
- `eig_f64_256x256` on `vmi1227854`: `[49.080 ms 49.892 ms 51.006 ms]`
- `eig_timing_probe` follow-up at n=1024 records `eig` as roughly 75% `eigvals` / serial Francis QR, with eigenvector machinery already parallelized.

Rejected shortcut families:

- Eigenvector-only back-substitution parallelism: rejected by same-worker probe in `22904991`.
- Eigvals column lower-bound active-window trim: bit-exact but no-op on the dense spectrum; `l` stays near zero.
- Lazy `q_acc` allocation for eigvals: does not target the measured QR floor and is below the score gate.
- Relaxed deflation threshold only: not acceptable as AED; risks changing eigenvalues without producing a window Schur proof.

## Canonical Mapping

The alien-graveyard corpus has no direct `dlaqr`/AED entry, but it does provide the relevant implementation class:

- Communication-avoiding linear algebra: reduce data movement and use dense BLAS-3 kernels for inner updates.
- Cache/locality and SIMD guidance: keep dense submatrix kernels contiguous and evidence-gated.

The alien-artifact numerical-linear-algebra contract supplies the proof shape:

- characterize the dense Hessenberg window,
- verify factorization/eigenvalue accuracy,
- preserve ordering and conjugate-pair semantics,
- record convergence/deflation evidence,
- expose fallback behavior.

## Recommendation Card

Change:
Implement an AED sub-window as a bounded trailing-window Schur transform for the active Hessenberg block, preserving the current double-shift path as strict fallback.

Hotspot evidence:
`eigvals_f64_256x256` median `27.445 ms` is the shared floor; n=1024 probe identifies the serial Francis QR as the O(n^3) wall.

Mapped graveyard sections:
Communication-avoiding linear algebra and dense matrix-kernel locality guidance. This is standard LAPACK `dlaqr2/3` methodology, not an exotic-corpus shortcut.

EV score:
`Impact 5 * Confidence 3 / Effort 5 = 3.0`, passes the `>= 2.0` gate only if the implementation attacks sweep count or enables multishift shifts. Diagnostic-only source changes do not pass.

Priority tier:
S/A for the geev no-gaps lane.

Budgeted mode:
Use a fixed AED window (`nw <= 32` initially), no dynamic heap growth beyond window work buffers, and no public dispatch if proof or same-worker gate fails.

Fallback trigger:
Any focused eig/eigvals test failure, golden strict-fallback digest loss, reconstruction/conjugate-pair drift beyond the ratified dense-eig tolerance, or same-worker `eigvals_f64_256x256` median non-win.

Isomorphism proof plan:

- Ordering: preserve current returned interleaved `(re, im)` order for strict fallback; any tolerance route must document block/order changes and keep `eigvals_matches_eig`.
- Tie/conjugate behavior: keep complex pairs adjacent and conjugate-signed.
- Floating point: AED/multishift may reassociate only under `qgce4` tolerance parity; strict fallback must remain bit-exact to the Pass 1 digests.
- RNG: none.
- Golden: verify n64/n128/n256 strict fallback digests and focused reconstruction tests after source changes.

p50/p95/p99 target:
Primary keep target is `eigvals_f64_256x256` same-worker median at least `1.25x` for an AED-only sub-lever, or route to `npxbw` if AED alone only prepares shifts. Combined fql10-D target remains `>=2.0x`.

Primary risk + countermeasure:
Risk: transforming a trailing Schur window spreads the Hessenberg spike and breaks the Hessenberg invariant. Countermeasure: no partial transform may be wired unless it also restores/deflates the spike or leaves the global matrix unchanged.

Repro artifact pack:
Use `pass1_baseline_profile.md`, `pass1_baseline_*.log`, `eig_timing_probe`, pre/post golden logs, focused eig tests, and same-worker Criterion after logs.

Rollback:
Remove the candidate source hunk and keep this contract plus rejected-route notes.

Baseline comparator:
Current `eig_francis_schur` double-shift scalar bulge chase in `crates/ft-kernel-cpu/src/lib.rs`.

## Selected Pass 3 Lever

Attempt exactly one bounded AED source lever:

1. Add a private, bounded AED window helper only if it can preserve the global Hessenberg invariant.
2. Do not change public eig/eigvals dispatch unless the helper is called from the active Francis loop and passes focused correctness tests.
3. If the helper cannot safely deflate or restore the spike, do not ship source; record the rejection and route to `npxbw` multishift scaffolding.
