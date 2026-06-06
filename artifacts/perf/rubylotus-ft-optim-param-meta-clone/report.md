# frankentorch-ooza ft-optim redundant metadata clone removal

Agent: RubyLotus
Target: `ft-optim` optimizer step paths
Crate: `ft-optim`

## Profile-backed target

After `frankentorch-bpow`, clean ft-data profiles showed no large next target. The next clean
profile-backed surface was `ft-optim` optimizer steps. `SGD::step` and sibling optimizers fetched
`session.tensor_values_meta(param)?.1.shape().to_vec()` immediately after cloning the parameter
values with `session.tensor_values(param)?`; the bound `_param_shape` was unused. That meant an
extra full parameter value clone per parameter with no arithmetic or validation effect on valid
optimizer inputs.

One lever: remove the unused `tensor_values_meta(param)` calls from the optimizer step paths that
already validated parameter existence and gradient length through `tensor_values(param)?`.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Worker: `vmi1149989`

Results:

```text
adamw/step_64x1024 [458.52 us 469.68 us 482.67 us]
adam/step_64x1024  [427.03 us 504.22 us 574.83 us]
sgd/step_64x1024   [245.43 us 283.24 us 333.89 us]
```

## After

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1149989 rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Worker: `vmi1149989`

Results:

```text
adamw/step_64x1024 [455.23 us 475.02 us 502.50 us]
adam/step_64x1024  [324.39 us 339.90 us 353.59 us]
sgd/step_64x1024   [225.12 us 238.00 us 250.24 us]
```

Attributable same-worker delta:

- `SGD::step` median: 283.24 us -> 238.00 us
- Speedup: 1.19x
- `AdamW::step` was effectively flat/noisy and is not claimed as a win for this lever.
- `Adam::step` moved faster in the after run, but this lever did not touch Adam's step path, so that
  movement is treated as benchmark noise and not counted.

Score: Impact 1.19 x Confidence 3.5 / Effort 1.0 = 4.2. Keep threshold passed.

## Isomorphism proof

- Arithmetic preserved: the removed calls only produced `_param_shape`, which was never read. All
  gradient, weight-decay, momentum, accumulator, and update expressions are unchanged.
- Ordering preserved: optimizer state updates and `apply_param_update` calls remain in the same
  order; no state write moved across a fallible session mutation.
- Floating point preserved: no floating-point operation was added, removed, or reordered.
- RNG/tie-breaking: optimizers in this path use no RNG and no tie-breaking.
- Error behavior for valid optimizer inputs is preserved. `session.tensor_values(param)?` still
  validates parameter lookup and materializes the values used for gradient-length checks; the removed
  metadata lookup was redundant after a successful value fetch.
- Golden outputs preserved: `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`
  passed for all present fixtures, including `ft_optim_adamw_pass17.txt` and
  `ft_optim_adamw_first_step_fused_frankentorch-wxtp.txt`.

## Verification

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-optim --all-targets
cargo fmt -p ft-optim --check
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-optim sgd -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-optim --all-targets --no-deps -- -D warnings
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-optim -- --nocapture
ubs crates/ft-optim/src/lib.rs artifacts/perf/rubylotus-ft-optim-param-meta-clone/report.md
```

The crate-specific commands passed. Dependency compilation continues to report an existing
`unused_mut` warning in `crates/ft-nn/src/lib.rs:4595`; this pass did not edit ft-nn.

UBS exited 1 on pre-existing heuristic findings in `ft-optim`, including false-positive secret
comparison reports on numeric and shape equality checks plus broad test-only `unwrap` and indexing
inventories. No UBS finding points at the removed metadata-clone lines.
