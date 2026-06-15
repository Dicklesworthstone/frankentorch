# frankentorch-4loqe - SGD borrowed-gradient update keep

## Target

- Bead: `frankentorch-4loqe`
- Crate: `ft-optim`
- Hot row: `sgd/step_64x1024`
- Lever: update SGD parameters through the borrowed accumulated-gradient closure so the step avoids cloning the gradient vector and avoids building a separate update vector.
- Profile source: prior optimizer report showed SGD near `213.32 us`; current source still cloned `grad` and allocated `update` for each parameter.

## Baseline

Command:

```bash
env RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 CARGO_TERM_COLOR=never rch exec -- cargo bench -j 1 -p ft-optim --bench optimizer_bench -- sgd/step_64x1024 --warm-up-time 1 --measurement-time 5 --sample-size 20 --noplot
```

RCH selected `vmi1227854`.

Criterion:

```text
sgd/step_64x1024 time: [244.89 us 252.25 us 260.31 us]
```

Evidence: `pass1_baseline_sgd_ts1.log`.

## Candidate

Command:

```bash
env RCH_REQUIRE_REMOTE=1 RCH_WORKERS=vmi1227854 CARGO_TERM_COLOR=never rch exec -- cargo bench -j 1 -p ft-optim --bench optimizer_bench -- sgd/step_64x1024 --warm-up-time 1 --measurement-time 5 --sample-size 20 --noplot
```

Criterion:

```text
sgd/step_64x1024 time: [215.77 us 228.64 us 240.70 us]
```

Evidence: `pass3_candidate_sgd_vmi1227854.log`.

## Delta

- Median: `252.25 us -> 228.64 us`
- Speedup: `1.103x`
- Score: `2.10` (`Impact 1.10 x Confidence 0.95 / Effort 0.50`)
- Verdict: KEEP

## Isomorphism Proof

The retained change preserves the observable SGD contract:

- Parameter iteration order is unchanged.
- Missing-gradient parameters are still skipped.
- Gradient/parameter length checks happen before mutation.
- `maximize` still negates the gradient before weight decay.
- Weight decay still uses the original parameter value for the current element.
- First-step momentum still seeds the velocity buffer with the effective gradient.
- Later momentum steps still use `momentum * v + (1 - dampening) * effective_grad`.
- Nesterov still applies `lr * (effective_grad + momentum * v)`.
- Vanilla SGD still applies `lr * effective_grad`.
- No RNG, tie-breaking, or cross-parameter scheduling behavior is introduced.

The only structural difference is storage movement: the gradient and parameter slices are consumed by the existing borrowed accumulated-gradient update API, eliminating the cloned gradient vector and the separate update vector.

## Gates

- `cargo test -j 1 -p ft-optim sgd -- --nocapture` on `vmi1227854`: passed, `29 passed; 0 failed`.
- `cargo check -j 1 -p ft-optim --all-targets` on `vmi1227854`: passed. A pre-existing dependent `ft-nn` `unused_mut` warning remains.
- `cargo clippy -j 1 -p ft-optim --lib --no-deps -- -D warnings` on `vmi1227854`: passed.
- `cargo clippy -j 1 -p ft-optim --all-targets --no-deps -- -D warnings`: blocked by a pre-existing test lint outside this SGD lever, `needless_range_loop` indexing `want_lr_lin`.
- `cargo fmt -p ft-optim -- --check`: blocked by pre-existing broad ft-optim formatting drift outside the touched SGD hunk.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`: passed for all present golden outputs.
- `git diff --check -- crates/ft-optim/src/lib.rs`: passed.
- `ubs crates/ft-optim/src/lib.rs`: completed with existing broad inventory; the report also records no unsafe blocks, no Tokio spawn usage, clean internal formatting/clippy/check/test-build summaries, and no unique blocker from this lever.

## Evidence Files

- `pass1_baseline_sgd_ts1.log`
- `pass2_test_ft_optim_sgd_vmi1227854.log`
- `pass3_candidate_sgd_vmi1227854.log`
- `pass4_check_ft_optim_vmi1227854.log`
- `pass5_clippy_ft_optim_nodeps_vmi1227854.log`
- `pass6_clippy_ft_optim_lib_nodeps_vmi1227854.log`
- `pass7_fmt_ft_optim_check.log`
- `pass8_golden_sha256_check.log`
- `pass9_ubs_ft_optim_src.log`
- `pass10_git_diff_check.log`
