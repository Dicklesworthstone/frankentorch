# frankentorch-t1vg pass 2 proof design

Date: 2026-06-07
Skill: `extreme-software-optimization`
Scope: read/review only. No source edits.

## Verdict

The current `functional_linear` hunk is behavior-isomorphic for the intended
profiled training path where `input`, `weight`, and optional `bias` are normal
leaf tensors requiring grad and are not mutated between forward and backward.
It maps the f64 grad custom op from owned `ctx.save_for_backward(x/w)` storage
to `tensor_apply_function_f64_borrowed_inputs` without changing the forward
kernel, backward kernel, input ordering, output shape, dtype, or gradient
ordering.

It is not yet fully proven across every allowed mutation/version state in this
repo. `BorrowedInputsF64` re-borrows current tape values during backward, and I
do not see a saved-version check in the borrowed-input dispatch. The repo
forbids in-place mutation of leaf tensors requiring grad, but that guard is not
a blanket prohibition for every non-leaf requiring-grad tensor. A keep commit
should therefore add a Linear-specific mutation-contract test, and ideally
either cover or explicitly reject the non-leaf in-place mutation case.

## Mapping

- `crates/ft-api/src/lib.rs:16147`: `functional_linear` f64 grad path now calls
  `tensor_apply_function_f64_borrowed_inputs`.
- `crates/ft-api/src/lib.rs:16149-16158`: forward still reads `ins[0]` as `x`,
  `ins[1]` as `w`, optional `ins[2]` as bias, calls
  `ft_kernel_cpu::linear_tensor_f64`, and derives output shape from input shape.
- `crates/ft-api/src/lib.rs:16160-16177`: backward reads
  `borrowed_inputs[0].0` and `[1].0`, calls
  `ft_kernel_cpu::linear_backward_f64`, then returns `[dx, dw, db?]`.
- `crates/ft-api/src/lib.rs:6428-6451`: API helper delegates to the tape helper.
- `crates/ft-autograd/src/lib.rs:8392-8477`: tape helper stores the same custom
  function input node IDs and registers `CustomFunctionBackward::BorrowedInputsF64`.
- `crates/ft-autograd/src/lib.rs:13526-13535`: backward reconstructs borrowed
  input slices from tape values before invoking the closure.

## Proof Obligations

- Ordering/tie: no sorting, ties, maps, random iteration, or set ordering are
  introduced. Input order remains `[input, weight, bias?]`; gradient order
  remains `[dx, dw, db?]`.
- Floating point: forward still uses `linear_tensor_f64`; backward still uses
  `linear_backward_f64`. The hunk does not change GEMM dimensions, loop order,
  bias accumulation, or `db=sum_batch(dy)` order. Proof should compare f64 bits
  or a golden digest for the benchmark-scale path.
- RNG: no RNG calls are added or removed. The benchmark creates random tensors
  before `functional_linear`; the hunk only changes custom-op saved storage.
- Output shape/dtype: output shape remains input shape with the last dimension
  replaced by `out_features`; borrowed helper creates an f64 output tensor.
- Bias/no-bias: no-bias path has two inputs and returns two gradients. Bias path
  has three inputs and returns `db` as the third gradient. Bias dtype/shape gates
  are unchanged before entering the fast path.
- Mutation contract: leaf tensors requiring grad are protected by
  `validate_tensor_in_place_target`, and existing conv2d precedent tests that
  contract. Linear still needs equivalent direct coverage. Non-leaf mutation
  needs explicit coverage because borrowed backward reads current values while
  the old owned-save path read forward-time copies.
- Golden output: keep proof must rerun
  `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`
  after the source cleanup/test addition and before commit.

## Existing Coverage

- `functional_linear_f64_grad_fused_matches_analytic` checks forward values and
  analytic `dx`, `dw`, and `db` for the f64 grad fast path.
- `functional_linear_one_dim_bias_broadcasts_and_sums_gradient` covers bias
  gradient summing for one simple shape.
- `custom_function_borrowed_inputs_backward_uses_tape_values` covers the raw
  autograd borrowed-input helper.
- `functional_conv2d_borrowed_grad_preserves_input_contract` is the closest
  precedent for mutation-contract coverage on this helper family.

## Recommended Missing Focused Tests

1. Add `functional_linear_borrowed_grad_preserves_input_contract`, mirroring the
   conv2d test: create leaf `input`, `weight`, `bias` requiring grad; run
   `functional_linear`; attempt `tensor_add_(weight, delta)` before backward and
   assert it fails; run backward and assert exact `dx`, `dw`, and `db`.
2. Add a no-bias f64 grad case that proves return arity `[dx, dw]` and exact
   gradients, since the current analytic test uses bias.
3. Add or decide on a non-leaf mutation test. If in-place mutation of a non-leaf
   input/weight is allowed by contract, the borrowed-input hunk is not
   isomorphic to owned save-for-backward. If it should be forbidden, add the
   guard/test before keeping this lever.
4. For the keep proof, add a bench-scale deterministic digest for
   `linear_train/hidden/2048` or an equivalent fixed-input Linear backward case
   so the golden sha captures output and gradients at realistic dimensions.

## Source Issues Before Clippy

- The pass-1 `ws` warning appears fixed in the live hunk: the current code uses
  `_ws` at `crates/ft-api/src/lib.rs:16151`.
- `git diff --check -- crates/ft-api/src/lib.rs` passed.
- I did not run clippy or tests in this pass. Before a keep commit, run focused
  crate-scoped RCH checks only, at minimum:
  `cargo test -p ft-api functional_linear`,
  `cargo test -p ft-autograd custom_function_borrowed_inputs_backward_uses_tape_values`,
  `cargo clippy -p ft-api --all-targets -- -D warnings`, and golden checksum
  verification.

