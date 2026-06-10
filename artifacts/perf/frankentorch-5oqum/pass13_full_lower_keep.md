# Pass 13 Full-Lower Staged TRED2 Keep

## State

- Worktree: `/data/projects/frankentorch-5oqum-boldfalcon`
- Bead: `frankentorch-5oqum`
- One lever: `eigvalsh_two_stage_f64` now keeps the stage-1 row-major band matrix
  and runs a private full-row-major lower-triangle values-only TRED2 reducer,
  instead of copying the band into packed lower storage first.
- Public dispatch: unchanged. `eigvalsh_contiguous_f64` and `eigh_contiguous_f64`
  still use the existing public paths.

## Profile Target

Pass12 private split on `vmi1227854` showed the staged residual is not QL:

| Stage | Time |
| --- | ---: |
| `stage1_values` | 6785.09 us/iter |
| `lower_pack` | 5.31 us/iter |
| `packed_tred2` | 6527.02 us/iter |
| `values_ql` | 270.85 us/iter |
| `two_stage_total` | 9458.28 us/iter |

That made the packed TRED2 memory layout a profile-backed target.

## Benchmark

Same-worker `vmi1227854`, Criterion, `cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- 'eigvalsh_f64_256x256|eigvalsh_two_stage_f64_256x256_b32' --warm-up-time 1 --measurement-time 2 --sample-size 10`.

| Row | Before | After |
| --- | ---: | ---: |
| `eigvalsh_two_stage_f64_256x256_b32` | 11.491 ms | 9.9628 ms |

Same-run public comparator after the change:

| Row | Median |
| --- | ---: |
| `eigvalsh_f64_256x256` | 6.5753 ms |

Public dispatch remains unchanged because the staged path is still slower than
the public path.

## Proof

- Bit-exact layout proof: `eigh_tred2_values_only_full_lower_matches_packed_bit_exact` passed on `vmi1227854`.
- End-to-end staged/live proof: `eigvalsh_two_stage_matches_live` passed on `vmi1227854`.
- Public golden SHA: `1870e56ea935f9cc895b24d878db52fe341dc2b195c00656faa38b2db97ac458`, unchanged.
- Ordering/ties: final output still sorts with `f64::total_cmp`.
- Floating point: full-row-major reducer mirrors the packed reducer's scale, dot,
  and update order; the proof compares `to_bits()` for `d`, `e`, and every lower
  row.
- RNG: none.

## Gates

- `cargo check -j 1 -p ft-kernel-cpu`: passed on `vmi1227854`.
- `cargo clippy -j 1 -p ft-kernel-cpu -- -D warnings`: passed on `vmi1227854`.
- `cargo fmt -p ft-kernel-cpu --check`: passed locally after RCH refused fmt as non-compilation.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: 0 critical issues; broad pre-existing warning inventory remains.

## Score

`(Impact 3 * Confidence 5) / Effort 2 = 7.5`; keep.

## Next Route

The remaining gap is not a QL/secular problem. Continue with true
compact-WY/dsytrd or stage1/TRED2 algorithmic replacement, and do not wire the
staged path into public dispatch until same-worker staged beats live.
