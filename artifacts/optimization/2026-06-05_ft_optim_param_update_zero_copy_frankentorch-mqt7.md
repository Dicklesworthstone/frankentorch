# ft-optim zero-copy parameter update keep - frankentorch-mqt7

Date: 2026-06-05
Bead: `frankentorch-mqt7`
Lever: safe closure-based f64 parameter mutation through `FrankenTorchSession`, routed through Adam and AdamW.

## Target

Profile-backed target: `ft-optim` `optimizer_bench` Adam/AdamW step path still cloned parameter values out of the session and wrote a full replacement `Vec` back after the previous fused-math pass. The remaining cost was session clone-out/clone-in round trips, not optimizer arithmetic.

## Implementation

- Added `DenseTensor::update_contiguous_values_with`.
- Added `TensorTape::values_len` and `TensorTape::update_tensor_values_with`.
- Added `FrankenTorchSession::tensor_values_len` and `tensor_update_param_values_f64_with`.
- Routed Adam and AdamW step loops through the closure API after all length/state validation.
- Kept AdamW state commit ordering: `step_counts`, `m`, and `v` are still written only after the session parameter mutation succeeds.

## Criterion

Command:

```text
RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.scratch/frankentorch-mqt7-baseline-target rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- adam --warm-up-time 1 --measurement-time 5 --sample-size 20
RCH_REQUIRE_REMOTE=1 CARGO_TARGET_DIR=/data/projects/.scratch/frankentorch-mqt7-after-target rch exec -- cargo bench -p ft-optim --bench optimizer_bench -- adam --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Same worker: `ts2`.

| Benchmark | Baseline p50 | After p50 | Ratio |
| --- | ---: | ---: | ---: |
| `adamw/step_64x1024` | `830.93 us` | `742.07 us` | `1.12x` |
| `adam/step_64x1024` | `511.54 us` | `467.42 us` | `1.09x` |
| combined p50 | `1342.47 us` | `1209.49 us` | `1.11x` |

Score: `(Impact 3.0 * Confidence 0.90) / Effort 1.10 = 2.45`, keep.

## Isomorphism

- Ordering/tie-breaking: no ordering operations changed.
- Floating point: Adam and AdamW per-element operation order is unchanged; only the storage target changes from a cloned `Vec` to the session-owned f64 slice.
- RNG: no RNG calls added, removed, or reordered.
- Error timing: Adam length/dtype validation still happens before state buffers are used; AdamW state-length checks happen before parameter mutation, preserving fail-closed optimizer state behavior.
- Golden outputs: `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed for all present fixtures, including `ft_optim_adamw_pass17.txt` and `ft_optim_adamw_first_step_fused_frankentorch-wxtp.txt`.

## Validation

- PASS: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-optim --all-targets`
- PASS: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-core dense_tensor_update_contiguous_values_with_mutates_in_place -- --nocapture`
- PASS: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-optim adam_first_step_matches_exact_float_bits -- --nocapture`
- PASS: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-optim adamw_first_step_matches_exact_decoupled_decay_values -- --nocapture`
- PASS: `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`
- PASS: `cargo fmt --check -p ft-core -p ft-optim`
- PASS: `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-optim --all-targets --no-deps -- -D warnings`
- PASS: `git diff --check -- crates/ft-core/src/lib.rs crates/ft-autograd/src/lib.rs crates/ft-api/src/lib.rs crates/ft-optim/src/lib.rs artifacts/optimization/2026-06-05_ft_optim_param_update_zero_copy_frankentorch-mqt7.md`
- REVIEWED: `ubs crates/ft-core/src/lib.rs crates/ft-optim/src/lib.rs artifacts/optimization/2026-06-05_ft_optim_param_update_zero_copy_frankentorch-mqt7.md` completed; reported existing heuristic debt, including false-positive "secret compare" hits on numeric length checks in `ft-optim` helpers, and no finding on the new closure API or Adam/AdamW mutation changes.

Known broader gate state:

- `cargo fmt --check -p ft-core -p ft-autograd -p ft-api -p ft-optim` fails on existing ft-api/ft-autograd formatting debt outside this lever.
- `cargo clippy -p ft-optim --all-targets -- -D warnings` fails while linting `ft-api` path dependency debt (`184` pre-existing ft-api lint errors); the optimizer crate itself passes with `--no-deps`.
- `ubs` over the full touched surface including `ft-api/src/lib.rs` and `ft-autograd/src/lib.rs` hung in an `ast-grep` shadow-workspace subscan and was terminated after several minutes; the smaller `ft-core`/`ft-optim`/artifact UBS scan completed as described above.
