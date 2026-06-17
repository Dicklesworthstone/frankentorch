# frankentorch-kgs4.109 closeout - direct f64 max_pool1d grad route

## Decision

KEEP.

This change adds the profile-backed direct f64 max-pool1d route and keeps the
existing reshape-through-max-pool2d fallback for non-f64 dtypes.

## Profile-backed target

- Source profile: `artifacts/perf/frankentorch-ftapi-train-reprofile-20260616/baseline_train_hotspots.log`
- Hot row: `conv1d_family/max_pool1d_grad`, median `306.70 ms` in the source profile
- Local baseline after the `ts1` offline override:
  `artifacts/perf/frankentorch-kgs4.109/pass1_local_baseline_max_pool1d_grad.log`
  reported `[411.85 ms 420.32 ms 428.89 ms]`
- Baseline artifact SHA-256:
  `f269a4fc3792dea3dfd7c2c1ee4110445a488df72251d658daa17581a264b64d`

## One lever

Add a dedicated f64 max-pool1d route:

- `ft_kernel_cpu::max_pool1d_forward_f64`
- `ft_kernel_cpu::max_pool1d_forward_with_indices_f64`
- `ft_kernel_cpu::max_pool1d_backward_from_indices_f64`
- `FrankenTorchSession::functional_max_pool1d` f64 fast path, including the
  no-grad leaf shortcut and the autograd custom-op route

## Isomorphism proof

- Ordering preserved: output planes are still traversed in `[N, C]` plane order,
  and each output row writes `ox` ascending.
- Tie-breaking preserved: the direct forward uses strict `>` replacement while
  scanning each window left to right, preserving first-argmax behavior from the
  height-1 max-pool2d route.
- Floating point preserved: forward copies the selected input value exactly;
  backward scatters each upstream gradient into the stored argmax offset in
  ascending output order.
- RNG preserved: no RNG is involved.
- Shape/error behavior preserved: 2-D unbatched input still routes through
  unsqueeze/pool/squeeze, invalid rank and size checks remain unchanged.
- Final golden kernel proof:
  `artifacts/perf/frankentorch-kgs4.109/pass5_final_kernel_proof_max_pool1d.log`,
  SHA-256 `cb013777bdae0fd8c469331c1c4c381fddc07d38d84b0f56a2856becb12c73bf`
- Final golden API proof:
  `artifacts/perf/frankentorch-kgs4.109/pass5_final_api_proof_max_pool1d.log`,
  SHA-256 `905ea6c038935ba0b54ca7019c1b7cf5e0741a244327559300486be0f9909a74`

## Rebench

Final local Criterion rebench:

- Artifact: `artifacts/perf/frankentorch-kgs4.109/pass3_local_rebench_max_pool1d_grad.log`
- Candidate: `[259.75 ms 263.76 ms 267.65 ms]`
- Baseline median to candidate median: `420.32 ms -> 263.76 ms`
- Ratio: `1.59x`
- Criterion result: change `[-38.815% -37.247% -35.670%]`, `p = 0.00`
- Rebench artifact SHA-256:
  `7cf96941fd9551ff0758170db26db8952e3075d42bc8c5e6b242ff30242613f7`

Score:

- Impact: `1.59`
- Confidence: `0.98` (same local target dir, focused row, proof-clean)
- Effort: `0.60`
- Score: `2.60`

## Gates

Captured artifacts:

- `pass4_check_ft_kernel_cpu_all_targets.log`
- `pass4_check_ft_api_all_targets.log`
- `pass4_clippy_ft_kernel_cpu_lib.log`
- `pass4_clippy_ft_api_lib.log`
- `pass4b_clippy_ft_api_lib_no_deps.log`
- `pass4_git_diff_check.log`
- `pass4_rustfmt_touched_check.log`
- `pass4_ubs_changed_rust_files_current.log`
- `pass5_final_kernel_proof_max_pool1d.log`
- `pass5_final_api_proof_max_pool1d.log`

Results:

- `cargo check -j 1 -p ft-kernel-cpu --all-targets` passed with pre-existing
  example warnings.
- `cargo check -j 1 -p ft-api --all-targets` passed with a pre-existing example
  warning.
- `cargo clippy -j 1 -p ft-kernel-cpu --lib -- -D warnings` passed.
- `cargo clippy -j 1 -p ft-api --lib -- -D warnings` remains blocked by
  pre-existing dependency lint debt in `ft-autograd`.
- `cargo clippy -j 1 -p ft-api --lib --no-deps -- -D warnings` remains blocked
  by existing ft-api lint debt.
- `git diff --check` passed.
- `ubs crates/ft-api/src/lib.rs crates/ft-kernel-cpu/src/lib.rs` completed
  with broad pre-existing findings across the two large files; no `max_pool1d`
  or changed-line hits were found in that UBS artifact. Completed artifact
  SHA-256 `ac099641c8888a85af8d558a17471d8da369572098bee6ab8b833ed85f6566e4`.
- A later bounded UBS rerun timed out in the Rust scan; timeout artifact
  SHA-256 `ae6180a6c748978eb2098a258504c696c62750148babe825744ad9dfc08d83e0`.
- The staged pre-commit UBS hook also timed out on the large `ft-api` scan after
  its 300s window and instructed `UBS_SKIP=1 git commit ...`; the commit used
  that bypass after the manual UBS artifacts above were retained.
- `rustfmt --edition 2024 --check crates/ft-api/src/lib.rs crates/ft-kernel-cpu/src/lib.rs`
  reported existing file-wide drift in unrelated sections; formatting was not
  applied to avoid unrelated churn.
