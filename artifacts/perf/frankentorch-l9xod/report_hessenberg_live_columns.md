# frankentorch-l9xod: eigvals Hessenberg Live-Column Left Apply

## Summary

Kept a one-lever eigvals-only optimization in `crates/ft-kernel-cpu/src/lib.rs`.

During Householder reduction to upper Hessenberg form, the eigvals path now skips
columns `< k` in the left reflector precompute/update. Those columns are outside
the active Hessenberg band and are not read by the Francis QR phase. Full
`eig` keeps the legacy full-column update so eigenvector behavior and timing are
unchanged by this lever.

## Benchmark Evidence

Primary paired A/B on `vmi1227854`, both with:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- eigvals_f64_256x256
```

| row | before | after | delta |
| --- | ---: | ---: | ---: |
| `eigvals_f64_256x256` | `[55.587 ms 57.542 ms 59.582 ms]` | `[53.037 ms 55.169 ms 57.415 ms]` | `1.043x`, `4.12%` lower median |

Earlier same-worker baseline, before the queue became noisy:

| row | baseline |
| --- | ---: |
| `eigvals_f64_256x256` | `[50.733 ms 52.334 ms 54.023 ms]` |
| `eig_f64_256x256` | `[123.69 ms 126.90 ms 130.39 ms]` |

Rejected probe evidence:

- Row-parallel final eigenvector back-transform regressed `eig_f64_256x256` from
  `[123.69 ms 126.90 ms 130.39 ms]` to `[129.28 ms 132.12 ms 135.12 ms]`.
- Applying the live-column skip to full `eig` produced a noisy `eig_f64_256x256`
  after row of `[130.10 ms 134.85 ms 140.40 ms]`, so the final kept hunk is
  intentionally scoped to `want_vectors == false`.

## Correctness Evidence

Remote final proof on `vmi1227854`:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo run -j 1 -p ft-kernel-cpu --release --example eigvals_golden
```

Golden digests stayed bit-exact:

```text
n=64  eigvals_digest=0xbc0583d464b1a211  eig_digest=0xbc0583d464b1a211
n=128 eigvals_digest=0xcf8084e9cc30d867  eig_digest=0xcf8084e9cc30d867
n=256 eigvals_digest=0x188d322a66b49c0f  eig_digest=0x188d322a66b49c0f
```

Additional final gates:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo test -j 1 -p ft-kernel-cpu eig_ -- --nocapture
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo check -j 1 -p ft-kernel-cpu --all-targets
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo clippy -j 1 -p ft-kernel-cpu --all-targets -- -D warnings
git diff --check -- crates/ft-kernel-cpu/src/lib.rs .skill-loop-progress.md .beads/issues.jsonl
ubs crates/ft-kernel-cpu/src/lib.rs
```

Results:

- Focused eig tests: `5 passed`.
- `cargo check`: passed.
- `cargo clippy`: passed.
- `git diff --check`: passed.
- UBS: `0` critical issues; warnings are the existing file-wide inventory.

## Isomorphism

- Ordering: eigvals QR ordering and deflation order are unchanged.
- Floating point: every live Hessenberg slot keeps the same dot-product order and
  update expression as before.
- RNG: no RNG is introduced.
- Tie handling: no eigenvalue ordering or tie policy changes.
- Full eig: `want_vectors == true` uses the previous full-column update.
- Golden output: eig/eigvals digest fixtures are unchanged for n=64, n=128, and n=256.

## Score

`Impact 2 * Confidence 4 / Effort 1 = 8.0`; kept.

Next deeper primitive for the remaining gap: multishift QR/AED, not another
row-parallel back-transform attempt.
