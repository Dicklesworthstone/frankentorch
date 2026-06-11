# frankentorch-ct2yy pass 4/5 keep: recursive QR panel factorization

## Target

- Bead: `frankentorch-ct2yy`
- Profile-backed hotspot: compact-WY QR panel factorization.
- Prior stage split: panel+T was `45.911%` for `512x512` and `68.787%` for `2048x128`.
- Alien primitive: recursive/block-recursive panel QR. Factor the first half of a panel, apply its compact-WY block reflector to the remaining panel columns with GEMM, then recurse.

## One Lever

Source hunk: `crates/ft-kernel-cpu/src/lib.rs`.

- Added `qr_factor_panel_recursive_f64`.
- Added `qr_apply_panel_block_reflector_f64`.
- Extracted `qr_build_compact_wy_t_f64`.
- Replaced the scalar within-panel reflector sweep in `qr_householder_panel_blocked_profiled` with the recursive panel factorizer.
- Public QR dispatch, output shapes, reduced/full mode routing, Householder sign policy, and `NB=32` are unchanged.

## Same-Worker Rebench

Worker: `vmi1227854`.

| Row | Baseline | Candidate | Speedup |
| --- | ---: | ---: | ---: |
| `qr_f64_512x512` | `67.182 ms` | `45.689 ms` | `1.47x` |
| `qr_f64_tall_2048x128` | `37.809 ms` | `25.196 ms` | `1.50x` |

Artifacts:

- Baseline square: `pass4_baseline_qr_512_vmi1227854_retry.log`
- Candidate square: `pass4_candidate_qr_512_any_remote.log`
- Tall baseline: `pass4_baseline_qr_tall_baseline_worktree_vmi1227854_retry.log`
- Candidate tall: `pass4_candidate_qr_tall_vmi1227854_retry.log`
- Supporting candidate tall rerun: `pass4_candidate_qr_tall_any_remote.log`

Score: `(Impact 4 * Confidence 5) / Effort 3 = 6.67`; keep.

## Isomorphism Proof

- Ordering/tie behavior: QR has no ordering or tie selection surface; reflector order stays left-to-right by panel and column.
- Floating point: the old compact-WY QR already used tolerance parity rather than bit-for-bit parity against the unblocked sweep. This lever additionally reassociates only the within-panel update from scalar per-column application to a compact-WY block update. That changes exact Q/R bits versus the previous blocked implementation, but preserves the QR contract through reconstruction and orthonormality checks.
- Householder sign policy: unchanged (`sign = if v0 >= 0.0 { 1.0 } else { -1.0 }`).
- RNG: none.
- Shapes/lower/upper semantics: unchanged by focused QR tests.
- Production/profile consistency: `qr_stage_split` reports `matches_production=true` for `512x512` and `2048x128` after the change.
- Post-change deterministic digests from the production QR output:
  - `512x512`: `532fcee14166c954`
  - `2048x128`: `22f5a5fdf69e511f`

Golden/proof bundle SHA-256: see
`artifacts/perf/frankentorch-ct2yy/pass4_proof_bundle.sha256`.

## Gates

- `rch exec -- cargo test -p ft-kernel-cpu --lib qr_ -- --nocapture`: pass, 16 tests.
- `rch exec -- cargo check -p ft-kernel-cpu --lib --tests --benches --example qr_stage_split`: pass.
- `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -A clippy::needless_range_loop -D warnings`: pass.
- `rch exec -- cargo clippy -p ft-kernel-cpu --lib --tests --benches --example qr_stage_split -- -D warnings`: fails on pre-existing `needless_range_loop` diagnostics; the detached parent worktree fails on the same library lint sites.
- `cargo fmt -p ft-kernel-cpu --check`: pass.
- `git diff --check -- crates/ft-kernel-cpu/src/lib.rs`: pass.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: 0 critical issues; pre-existing broad warnings remain in the large file.

Standard all-target clippy with `-D warnings` is still blocked by unrelated
pre-existing `needless_range_loop` diagnostics outside this QR hunk.

## Next Profile Target

After this keep, the square row's profile shifts toward reverse Q formation and the tall row still has a substantial panel share. The next deeper QR primitive should be one of:

- cache-blocked reverse `dorgqr`/Q formation, if `qr_stage_split` keeps showing reverse Q as dominant;
- TSQR/CAQR panel tree for tall QR, if tall panel cost remains dominant after reprofile;
- a tighter recursive-panel workspace strategy only after profiling proves allocation/copy cost has become material.
