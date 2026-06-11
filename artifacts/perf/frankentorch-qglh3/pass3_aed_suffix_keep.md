# frankentorch-qglh3 pass 3 AED suffix keep

Date: 2026-06-11
Agent: `IvoryDeer`
Bead: `frankentorch-qglh3`

## Target

Profile-backed residual: values-only non-symmetric Francis QR in
`eigvals_f64_256x256`.

Same-worker baseline on `vmi1227854`:

| Row | Estimate |
| --- | ---: |
| `eigvals_f64_256x256` | `[26.514 ms 27.445 ms 29.029 ms]` |

## One Lever

Added a conservative values-only AED suffix deflation probe inside
`eig_francis_schur`. On every fourth sweep, the probe copies a trailing 16x16
Hessenberg window, recursively Schur-factors that window with the extracted
`eig_francis_schur`, and deflates the suffix only when the window's spike vector
bound stays below the threshold.

This pass intentionally does not touch the `want_vectors` path. Full q_acc
back-transform and shift-list handoff remain the next qglh3 work.

## Behavior Proof

- Ordering/tie behavior: preserved through the existing interleaved `(re, im)`
  return path and `eigvals_matches_eig` focused test.
- Floating point: public strict golden digests unchanged for the existing n64,
  n128, and n256 fixtures.
- RNG: none.
- Golden digests from remote RCH `vmi1227854`:
  - n64: `0xbc0583d464b1a211`
  - n128: `0x763c4b15d92c4b89`
  - n256: `0x00b87b4996340204`

## Rebench

Same-worker after row on `vmi1227854`:

| Row | Estimate |
| --- | ---: |
| `eigvals_f64_256x256` | `[25.089 ms 25.741 ms 26.441 ms]` |

Median delta: `27.445 ms -> 25.741 ms`, `1.066x`.

Score: `(Impact 2 * Confidence 3) / Effort 3 = 2.0`; keep.

## Gates

- `env RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -j 1 -p ft-kernel-cpu eigvals -- --nocapture`
- `env RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -j 1 -p ft-kernel-cpu --all-targets`
- `env RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -j 1 -p ft-kernel-cpu --all-targets -- -D warnings`
- `cargo fmt -p ft-kernel-cpu --check`
- `ubs crates/ft-kernel-cpu/src/lib.rs`

UBS found no critical issues. It still reports pre-existing warnings in the
large kernel file; the AED hunk did not introduce a critical finding.

## Next Route

Continue qglh3 with full AED data plumbing: q_acc/window back-transform and
undeflated shift-list handoff for `frankentorch-npxbw` multishift sweeps. Do not
repeat threshold-only values tweaks.
