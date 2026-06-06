# frankentorch-8xxl AdamW in-place state update

## Target

- Bead: `frankentorch-8xxl`
- Crate: `ft-optim`
- Profile-backed hotspot: `adamw/step_64x1024`
- Lever: remove per-step `next_m`/`next_v` allocations in `AdamW::step`

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- adamw/step_64x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10
```

Worker: `ts1`

Result:

- `adamw/step_64x1024`: `[492.19 us 493.99 us 496.12 us]`

## Change

One lever:

- Replace locally allocated `next_m` / `next_v` buffers with direct updates to the persistent `self.m[i]` and `self.v[i]` buffers.
- Initialize missing state buffers inside `tensor_update_param_values_f64_with`, after tensor storage/layout validation has succeeded.
- Keep the exact AdamW arithmetic expression order for each element.

## After

Primary same-worker confirmation:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- adamw/step_64x1024 --warm-up-time 1 --measurement-time 5 --sample-size 10
```

Worker: `ts1`

Result:

- `adamw/step_64x1024`: `[293.90 us 298.34 us 303.34 us]`

Additional cross-worker checks:

- `vmi1149989`: `[334.18 us 342.72 us 357.69 us]`
- `vmi1149989`: `[331.44 us 345.63 us 355.83 us]`

Speedup:

- Same-worker median: `493.99 us -> 298.34 us`
- Ratio: `1.66x`

Score:

- `Impact 1.66 x Confidence 0.95 / Effort 0.7 = 2.25`
- Verdict: keep

## Isomorphism proof

- Ordering: parameter iteration, gradient lookup, step-count check, state-length validation, tensor update, and final `step_counts[i] = t` order are preserved.
- Floating point: each element still computes `m = beta1*m + (1-beta1)*g`, `v = beta2*v + (1-beta2)*g*g`, bias corrections, Adam delta, decoupled decay delta, then `p -= decay + adam_delta` in the same order.
- State behavior: missing state buffers are initialized only inside the non-fallible parameter-update closure. If tensor storage/layout validation fails before the closure, optimizer state is not initialized or advanced.
- RNG/tie-breaking: AdamW uses no RNG and no tie-breaking.
- Golden SHA-256: `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed, including `ft_optim_adamw_pass17.txt` and `ft_optim_adamw_first_step_fused_frankentorch-wxtp.txt`.

## Gates

Passed:

- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo check -p ft-optim --all-targets`
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-optim adamw -- --nocapture`
- `cargo fmt -p ft-optim -- --check`
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-optim --all-targets --no-deps -- -D warnings`
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`
- `git diff --check`

Blocked by pre-existing unrelated repo debt:

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-optim --all-targets -- -D warnings` fails in dependent `ft-api` pre-existing lints.
- `ubs crates/ft-optim/src/lib.rs` exits nonzero on broad pre-existing heuristic inventories. No finding is specific to this AdamW state-buffer lever.

## Next primitive

Re-profile after this keep. If the optimizer path remains the best isolated non-owned target, attack a deeper optimizer primitive such as persistent gradient borrow/update APIs to remove the remaining gradient clone, rather than another spelling-level AdamW tweak.
