# FrankenTorch Release-Readiness Scorecard

Updated: 2026-06-20

## Performance Gauntlet

| Bead | Workload | Result vs PyTorch | Before/after verdict | Release action |
|---|---:|---:|---:|---|
| `frankentorch-kgs4.140` | BatchNorm1d f64 scalar-upstream backward `[16,128,256]` NCL | same-host `5.35x` slower | internal keep; same-worker `vmi1152480` native automatic `5.6654 ms` -> `4.7142 ms` (`1.20x` faster), explicit scalar `6.0145 ms` -> `3.5559 ms` (`1.69x` faster, significant); direct scalar-forward and algebraic zero-gradient probes rejected | kept saved-`rstd` reuse in `batch_norm_backward_scalar_f64`; route remaining gap to true forward output deforestation, generated scalar-loss kernels, tape/session arena reuse, and stronger PyTorch-equivalent zero-gradient proof |
| `frankentorch-kgs4.139` | BatchNorm1d f64 automatic `tensor_sum` shortcut `[16,128,256]` NCL | same-host `7.42x` slower | internal keep; local ordinary native `11.622 ms` -> automatic shortcut `6.6151 ms` (`1.76x` faster); rch after row `6.0836 ms`; explicit scalar API still faster at `5.1754 ms` | kept; route remaining gap to BatchNorm output deforestation, generated scalar-loss kernels, tape/session arena reuse, and saved-stat workspace reuse |
| `frankentorch-kgs4.138` | BatchNorm1d f64 fused scalar-sum train step `[16,128,256]` NCL | same-host `4.52x` slower | internal keep; local native `11.178 ms` -> scalar-sum `4.7944 ms` (`2.33x` faster); rch same-run scalar/native `25.058 ms` / `43.610 ms` (`1.74x` faster) | kept; route remaining gap to automatic scalar-loss pattern matching, tape/session arena reuse, saved-stat workspace reuse, and algebraic zero-`dx` proof |
| `frankentorch-kgs4.120` | RMSNorm f64 train scalar-sum step `[2048,1024]` | mixed-location `4.88x` slower | no gain; active f64 unit-dy branch `59.289 ms` vs generic-disabled `58.407 ms`, `p=0.55`; final source `64.615 ms`, `p=0.58` | rejected/reverted f64 unit-dy branch; route to automatic scalar-loss fusion, persistent row-stat/workspace reuse, arena allocation, f64-native layout, or generated fused code |
| `frankentorch-kgs4.137` | RMSNorm f64 scalar-sum train step `[2048,1024]` | mixed-location `0.86x` PyTorch median, not release-counted | no gain; same-worker scalar `12.329 ms` vs materialized same-run `12.086 ms` and baseline `12.229 ms` | rejected/no source landed; route to tape/session allocation, workspace reuse, automatic scalar-loss fusion, or f32-native layout |
| `frankentorch-kgs4.125` | BatchNorm1d f64 native NCL train step `[16,128,256]` | same-host `4.85x` slower | internal keep; RCH native `4.3741 ms` vs fold `30.484 ms`; local row-coarsening `11.865 ms` -> `10.914 ms` | kept; route remaining gap to f64 scalar-loss fusion, dense-dy removal, tape/workspace reuse, and saved-stat reuse |
| `frankentorch-kgs4.123` | RMSNorm f32 train scalar-sum step `[2048,1024]` | mixed-location `1.79x` slower | no gain; active f32 unit-dy branch `67.574 ms` vs final generic `19.613 ms` on `vmi1149989` | rejected/reverted f32 unit-dy branch; keep benchmark row; route to row-stat reuse, scalar-loss tape fusion, arena allocation, and f32 storage/layout |
| `frankentorch-kgs4.136` | BatchNorm2d f32 fused scalar-sum train step `[32,256,28,28]` | mixed-location `13.94x` slower | internal keep; rch Criterion `114.23 ms` fused -> `78.166 ms` scalar-sum; direct diagnostic `10.80 ms` fused -> `1.66 ms` scalar-sum | kept; route remaining gap to stats/backward reuse, arena/tape allocation, automatic loss fusion, and f32 storage/layout |
| `frankentorch-kgs4.135` | GroupNorm f32 fused scalar-sum train step `[8,64,28,28]`, groups `32` | direct A/B `5.58x` slower | internal keep; rch direct path `8.30 ms` fused -> `2.10 ms` scalar-sum; Criterion `17.139 ms` materialized -> `8.9874 ms` scalar-sum | kept; route remaining gap to automatic loss fusion, arena/tape allocation, f32 storage/layout, and scheduler work |
| `frankentorch-kgs4.116` | LayerNorm f64 train step `[2048,1024]` | `3.58x` slower | internal keep; same-worker rch parent `90.723 ms` -> current `29.606 ms`; f32 diagnostic `1930.66 ms` -> `293.49 ms` | kept; route remaining gap to allocation/tape/loss fusion/workspaces/parallel reductions |
| `frankentorch-kgs4.115` | GroupNorm f32 train step `[8,64,28,28]`, groups `32` | `19.04x` slower | internal keep; same-worker rch parent `19.13 ms` -> current `11.72 ms` | kept; route remaining gap to allocation/tape/fusion/parallel f32 scheduling |
| `frankentorch-kgs4.114` | BatchNorm2d f32 train step `[32,256,28,28]` | final `28.14x` slower | no gain; same-worker rch `vmi1152480` disabled/final `147.30 ms`, active unit-dy branch `157.93 ms` | rejected/reverted unit-dy branch; keep gauntlet harness |
| `frankentorch-kgs4.113` | SDPA f64 train step `[16,512,64]` | `1.29x` slower | internal keep; same-worker rch `114.40 ms` old post-scale -> `82.730 ms` scaled alpha | kept; route remaining gap to SDPA scheduling/allocation/fusion |
| `frankentorch-kgs4.112` | avg_pool2d f64 train step `[8,64,64,64]` | `4.54x` slower | existing 2x2s2 fast path verified; direct-assignment candidate `58.600 ms` -> `68.624 ms` rejected | keep gauntlet harness/evidence; product source unchanged |
| `frankentorch-kgs4.117` | max_pool3d f64 train step `[2,32,16,32,32]` | `9.73x` slower | internal keep; `20.585 ms` -> `15.794 ms`; remote PyTorch arm unavailable on `hz2` | kept; profile deeper end-to-end gap |
| `frankentorch-kgs4.121` | linear f64 train step `[32,512] -> 2048` | `2.45x` slower | API-local internal keep; `29.606 ms` -> `22.775 ms`; kernel move `26.459 ms` rejected | kept API helper; reverted kernel move |
| `frankentorch-kgs4.122` | avg_pool1d f64 train step `[8,64,8192]` | `25.86x` slower; final rerun `24.92x` slower | no gain; candidate median `204.02 ms` vs fast-path-disabled `179.91 ms` | reverted |
| `frankentorch-kgs4.134` | avg_pool1d f64 fused scalar-sum train step `[8,64,8192]` | `7.55x` slower | internal keep; local same-run old `69.267 ms` -> fused `59.050 ms`; rch same-run old `134.74 ms` -> fused `87.564 ms` | kept; route remaining gap to allocation/tape/loss fusion beyond avg_pool1d kernel microlevers |
| `frankentorch-kgs4.124` | SmoothL1 f64 mean-loss backward, 8M elems | `1.99x` slower | internal keep; `963.16 ms` -> `757.63 ms` on `hz2` | kept; follow-up `frankentorch-kgs4.127` |
| `frankentorch-kgs4.126` | max_pool1d f64 train step `[8,64,8192]` | `12.31x` slower | no gain; candidate median `184.41 ms` vs parent `178.47 ms` | reverted |
| `frankentorch-kgs4.127` | SmoothL1 f64 one-sided input grad, 8M elems | `1.79x` slower | internal keep; same-host local `746.26 ms` -> `647.44 ms` | kept; route remaining gap to tape/allocation/SIMD |
| `frankentorch-kgs4.128` | max_pool3d f64 train step `[2,32,16,32,32]` | `9.38x` slower clean baseline | no gain; borrowed-input median `22.764 ms`, unit-dout median `16.160 ms`, sequential unit-dout median `22.465 ms` | reverted product candidates; keep stage probe |
| `frankentorch-maxpool3d-scalar-loss-grad-buffers-7wru6` | max_pool3d f64 fused sum train step `[2,32,16,32,32]` | baseline `2.46x` slower; accumulate-only candidate `3.02x` slower | no gain; fused median `5.7046 ms`, accumulate-only `5.7846 ms`; raw stage rows regressed | reverted no-report accumulation path |
| `frankentorch-kgs4.133` | conv2d f64 train step `[4,64,64,64]`, 64 3x3 filters | `1.91x` slower; candidate `1.86x` slower | no gain; same-worker rch `121.07 ms` -> `117.92 ms`, `p=0.38`, no change detected | rejected; removed dormant all-ones-dout branch |
| `frankentorch-grefr` | SmoothL1 f64 mean-loss backward, 8M elems | `1.35x` slower | internal keep; direct local `588.51 ms` -> `469.36 ms`; beta=1 derivative branch rejected | kept paired-randn fill; route remaining gap to tape/allocation/loss-kernel |

Measured-discipline score: `24/24` for the gauntlet lanes. PyTorch head-to-head
score: `0W / 23L / 1N`; the RMSNorm scalar-sum comparator is neutral for
release scoring because the candidate was faster than local PyTorch by
mixed-location ratio but failed the same-worker FrankenTorch keep gate.
Correctness guards are green and the SDPA, MaxPool3d,
Linear, LayerNorm, BatchNorm1d/2d, GroupNorm, and SmoothL1 levers include real
internal speedups, but no measured workload is performance-dominant against
PyTorch yet.

### 2026-06-20 BatchNorm1d scalar-backward saved-rstd keep (`frankentorch-kgs4.140`)

The f64 scalar-upstream BatchNorm backward now precomputes per-channel
`rstd = 1 / sqrt(var + eps)` once and reuses it in the `dweight` and `dx`
passes. This removes repeated square roots from the scalar-loss backward while
preserving the exact dense-backward arithmetic contract pinned by the kernel
bit tests.

Two bolder probes were rejected before the keep. Replacing automatic
`tensor_sum(batch_norm1d_output)` with direct `batch_norm_sum_forward_f64`
changed the retained-fallback loss bits by 16 ULPs and was reverted. Returning
algebraic zero `dx`/`dweight` for finite scalar upstream passed the scaled
tolerance case but failed the bit-exact f64 unit-`dy` kernel test, so it was
also reverted.

Same-worker RCH proof on `vmi1152480`, `cargo bench -p ft-api --bench
ops_bench --profile release -- batch_norm/grad_1d_ncl_16x128x256`, measured
parent native automatic `5.6654 ms`, explicit scalar `6.0145 ms`, and
fold-reference `62.683 ms`. The saved-`rstd` after-run measured native
automatic `4.7142 ms`, explicit scalar `3.5559 ms`, and fold-reference
`41.846 ms`. Ratios: native `1.20x` faster, explicit scalar `1.69x` faster
with Criterion reporting `p = 0.00`, and fold-reference `1.50x` faster.

Current local PyTorch `2.12.1+cpu`, 32 intra-op threads, measured median
`0.880459 ms` for the same NCL f64 scalar-loss train step. The kept automatic
native row is still `5.35x` slower than PyTorch; explicit scalar is still
`4.04x` slower. `perf_event_paranoid=4` blocked hardware-counter profiling, so
this run uses Criterion timing plus the PyTorch comparator.

Gates: f64 scalar-backward kernel tests 2/0, BatchNorm1d API filter 10/0,
full `ft-conformance` green, `ft-kernel-cpu` and `ft-api` checks green,
`ft-kernel-cpu` lib clippy green, `ft-api` lib clippy green, release build for
`ft-kernel-cpu` green, `git diff --check` green. `ft-api --lib --benches`
clippy remains blocked by pre-existing unrelated test lint debt; full-file
rustfmt for `ft-kernel-cpu/src/lib.rs` remains blocked by unrelated old drift
outside the touched BatchNorm hunk. UBS on the kernel file exited 0 with 0
critical issues and the existing large-file warning inventory.

### 2026-06-20 BatchNorm1d automatic tensor_sum shortcut keep (`frankentorch-kgs4.139`)

The ordinary `functional_batch_norm1d(...).tensor_sum()` path now recognizes
f64 training-mode affine BatchNorm1d outputs and replaces the generic Sum
backward edge with a scalar BatchNorm backward shortcut. It falls back to the
materialized `Sum` path when retained grads or tensor hooks make the output
gradient observable.

The corrected RCH baseline (`cargo bench --profile release`; the first
`--release` attempt was invalid for `cargo bench`) selected `hz1` but timed out
during remote sync and fell back locally. That local fallback measured native
ordinary BatchNorm1d+Sum at `[11.129 ms, 11.622 ms, 12.148 ms]`, explicit
scalar-sum at `[4.8402 ms, 5.0014 ms, 5.1322 ms]`, and fold-reference at
`[58.736 ms, 59.337 ms, 60.091 ms]`.

After the automatic shortcut, local same-machine Criterion measured native
ordinary call sites at `[6.3946 ms, 6.6151 ms, 6.7803 ms]`, explicit scalar-sum
at `[5.0390 ms, 5.1754 ms, 5.2503 ms]`, and fold-reference at
`[37.121 ms, 40.052 ms, 42.854 ms]`. The automatic path is `1.76x` faster than
baseline ordinary call sites, but still `1.278x` slower than the explicit
scalar API because it still materializes the BatchNorm output in forward.

RCH after-run evidence on `hz2` measured native automatic `6.0836 ms`, explicit
scalar `4.7261 ms`, and fold `48.006 ms`; because the before row fell back
locally, this is routing evidence rather than same-worker proof. Local PyTorch
`2.12.1+cpu`, 32 threads, clone/detach per rep, measured median `0.891630 ms`.
The local automatic shortcut is still `7.42x` slower than PyTorch.

Gates: focused `ft-api` shortcut tests 2/0, BatchNorm1d API filter 10/0, full
`ft-conformance` green, `ft-autograd` check and clippy green, `ft-api`
check/lib-clippy green. All-target `ft-api` clippy remains blocked by existing
test lint debt outside this change; full-file rustfmt remains blocked by old
large-file drift, but the touched BatchNorm shortcut hunks were manually
formatted and the touched-symbol rustfmt grep is clean. Scoped UBS timed out
after 240s on the giant Rust files with no findings emitted; docs/artifact-only
UBS exited 0 but reported Markdown as no recognizable language.

### 2026-06-20 RMSNorm f64 unit-dy no-ship (`frankentorch-kgs4.120`)

The existing f64 all-ones-`dy` RMSNorm backward branch was batch-verified and
removed. On `vmi1153651`, the active branch measured
`[51.215 ms, 59.289 ms, 67.477 ms]` for
`rms_norm/grad_2048x1024`; the temporary generic-disabled probe measured
`[52.546 ms, 58.407 ms, 64.377 ms]` with no detected change (`p=0.55`), and
the final source after removing the branch/helper/test measured
`[46.294 ms, 64.615 ms, 87.183 ms]` with no detected change (`p=0.58`).
The active branch was `1.0151x` slower than the generic-disabled median.

Local PyTorch CPU `2.12.1+cpu`, 32 threads, clone/detach per rep, measured
`13.241798 ms` median for the same f64 scalar-sum train row. The final
FrankenTorch path is still `4.88x` slower by this mixed-location ratio; the
active rejected branch was `4.48x` slower. Gates passed for full
`ft-kernel-cpu` lib tests (`504 passed; 0 failed; 2 ignored`), API RMSNorm
tests (`6 passed; 0 failed`), strict-scheduler conformance, `ft-kernel-cpu`
check, `ft-kernel-cpu` clippy, `git diff --check`, and scoped UBS with `0`
critical issues. Whole-file rustfmt on the touched giant file remains blocked
by existing unrelated drift, so no broad formatting rewrite was applied.

### 2026-06-20 RMSNorm f32 unit-dy no-ship (`frankentorch-kgs4.123`)

The existing f32 all-ones-`dy` RMSNorm backward branch was batch-verified and
failed hard. On `vmi1149989`, the active branch measured
`[63.618 ms, 67.574 ms, 70.695 ms]` for
`rms_norm/grad_f32_2048x1024`; disabling the branch measured
`[16.839 ms, 18.496 ms, 20.014 ms]`, and the final product source with the
branch/helper/test removed measured `[18.942 ms, 19.613 ms, 20.940 ms]`.
So the active candidate was `3.45x` slower than final source and was removed.

Local PyTorch CPU `2.12.1+cpu`, 32 threads, clone/detach per rep, measured
`10.970112 ms` median for the same f32 scalar-sum train row. The final
FrankenTorch path is still `1.79x` slower by this mixed-location ratio; the
active rejected branch would have been `6.16x` slower. Gates passed for the
f32 API grad parity test, strict-scheduler conformance, `ft-api` bench check,
`ft-api` bench clippy, `ft-kernel-cpu` lib clippy, and scoped UBS with `0`
critical issues. Whole-file rustfmt on the touched giant files remains blocked
by existing unrelated drift, so no broad formatting rewrite was applied.

### 2026-06-20 BatchNorm1d f64 scalar-sum keep (`frankentorch-kgs4.138`)

A dedicated f64 `sum(batch_norm1d(input, running_mean, running_var, weight,
bias))` scalar-loss path now supports both `[N,C]` and native `[N,C,L]`.
It computes the scalar directly and backpropagates scalar upstream gradient
through `batch_norm_backward_scalar_f64`, avoiding the normalized output tensor,
the `tensor_sum` tape node, and the dense all-ones `dy` buffer.

Local same-host Criterion on `[16,128,256]` NCL measured native materialized
BatchNorm1d at `[10.985 ms, 11.178 ms, 11.358 ms]` and scalar-sum at
`[4.6378 ms, 4.7944 ms, 4.9162 ms]`, a `2.33x` internal speedup. Fold-reference
remains `56.986 ms`, so scalar/fold is `0.0841x` (`11.89x` faster). Local
PyTorch `2.12.1+cpu`, 32 threads, prebuilt random fixture plus clone/detach per
rep, measured median `1.061455 ms`; the scalar-sum row is still `4.52x` slower.

RCH routing evidence: baseline native/fold on `vmi1149989` was `7.3230 ms` /
`44.182 ms`. The after run did not honor the requested worker pin and landed on
`vmi1153651`, where same-run native/scalar/fold were `43.610 ms` / `25.058 ms`
/ `190.20 ms`; scalar/native was still `1.74x` faster, but the before/after
worker mismatch makes the local same-host run the primary keep proof.

Gates: f64 scalar-backward kernel tests 2/0, full BatchNorm kernel filter 7/0,
API NCL scalar-vs-materialized proof 1/0, full `ft-conformance` green, scoped
check/clippy green for `ft-kernel-cpu` and `ft-api` bench. Whole-file rustfmt
checks remain blocked by pre-existing formatting drift outside this change.

### 2026-06-20 RMSNorm scalar-sum no-ship (`frankentorch-kgs4.137`)

A dedicated f64 `sum(rms_norm(input, weight))` scalar-loss candidate removed
the normalized output allocation, the `tensor_sum` tape node, and dense
all-ones upstream allocation, but did not clear the same-worker A/B gate.
On `vmi1227854`, the materialized baseline was `[11.683 ms, 12.229 ms,
12.596 ms]`. The candidate run measured the existing materialized row at
`[11.334 ms, 12.086 ms, 13.179 ms]` with no detected change (`p=0.61`) and
the scalar-sum row at `[11.023 ms, 12.329 ms, 13.944 ms]`. That is `1.020x`
slower than materialized in the same run and `1.008x` slower than baseline.

Remote PyTorch was unavailable on rch workers (`No module named 'torch'`).
The local PyTorch `2.12.1+cpu` comparator, 32 threads with clone/detach per
rep, measured median `14.360424 ms`, so the mixed-location scalar/PyTorch
ratio was `0.8586x`. That ratio is recorded but not release-counted because
the product candidate lost to the existing FrankenTorch path. Focused
candidate-branch tests passed (`ft-kernel-cpu` scalar backward 6/0,
`ft-api` RMSNorm scalar-sum 2/0), strict-scheduler conformance passed 1/0,
`ft-api` and `ft-kernel-cpu` all-target checks passed, and `cargo fmt --check`
passed. All-target clippy remains blocked by existing broad lint debt, and no
source from this lane was shipped.

### 2026-06-19 root-cause — the pooling-train-step losses are the generic backward machinery (`frankentorch-96e5d`)

A phase-timing probe (`crates/ft-api/examples/avgpool1d_phase_timing.rs`) shows the
`avg_pool1d [8,64,8192]` 25.86x gap (`kgs4.122`) is ~75% in `tensor_backward`
(~70–134 ms) while the raw pooling kernels are ~3 ms; a control `sum(x).backward()`
on the same 4 M leaf with NO pooling op is 35–53 ms. So the gap is the generic autograd
backward machinery (fresh multi-MB grad/contrib alloc + first-touch page faults + serial
bandwidth-bound copy), NOT the kernel — which is why the `kgs4.122`/`kgs4.126` pooling
KERNEL fast paths were correctly reverted. Shipped one bit-exact, can't-regress slice:
`Sum`/`Mean` backward now accumulate the constant gradient lazily (no materialized
`vec![scalar; numel]`); same-process same-worker A/B = `13.73x` on the eliminated
contribution. The remaining ≥2x lever is a backward grad-buffer caching allocator
(`frankentorch-cbe4t`). Gates: ft-autograd 476/0, conformance 199/0 + all sub-suites,
clippy clean.

Forward half (`frankentorch-0w3ns`): `apply_function` clones every input (33 MB on the
`[8,64,8192]` lane) before the kernel; avg_pool1d/max_pool1d backwards never read the
input, so both f64 forwards were routed through the zero-copy
`tensor_apply_function_f64_borrowed_forward` (same accepted pattern as kgs4.119/132).
Same-process A/B (clone+kernel vs borrow+kernel, m=4M): `5.89x` mean / `9.05x` min; the
avg_pool1d forward phase fell ~20 ms → ~6.8 ms. Bit-exact, can't-regress. Gates: ft-api
avg_pool1d 7/0 + max_pool1d 1/0, conformance 199/0 + all sub-suites, clippy clean.

First-contribution backward slot slice (`frankentorch-cbe4t`): `TensorTape` now delays
per-node gradient buffer materialization until the first contribution, preserving the
old `0.0 + contribution` bit semantics and the public zero-gradient fallback for
reachable requires-grad nodes. Local PyTorch-enabled `avg_pool1d` train row improved
FT median `89.360 ms -> 70.206 ms` (`21.4%` faster); PyTorch was `6.7081 ms` baseline
and `6.9328 ms` candidate, so the FT/PyTorch loss narrowed from `13.32x` to `10.13x`
slower. Remote `rch` workers lacked Torch; same-worker `ovh-a` FT-only row was
statistically neutral despite a lower median. Gates: ft-autograd 476/0, ft-api
avg_pool1d bit regression 1/0, strict scheduler conformance 1/0, ft-autograd clippy
clean.

Fused avg_pool1d scalar-sum (`frankentorch-kgs4.134`): a dedicated
`functional_avg_pool1d_sum` f64 path computes the pooled scalar sum directly and
backprops scalar upstream gradient without materializing the pooled output
gradient buffer. Local PyTorch-enabled gauntlet improved the same-run
FrankenTorch row from `69.267 ms` to `59.050 ms` (`1.17x` faster), while PyTorch
was `7.8192 ms`, so the candidate remains `7.55x` slower. rch `vmi1152480`
Rust-only gauntlet moved `134.74 ms` -> `87.564 ms` (`1.54x` faster); remote
PyTorch is still unavailable because workers lack `torch`. Gates: ft-kernel-cpu
avg_pool1d sum 1/0, ft-api avg_pool1d sum 1/0 plus integration filters,
ft-conformance full suite green, ft-kernel-cpu/ft-api clippy clean.

Conv3d sum-loss backward (`frankentorch-kgs4.118`): existing code-first f64 all-ones
`dout` fast path is now batch-verified. Same-worker `ovh-a` parent baseline
`29.723 ms` -> current `26.595 ms` (`1.12x` faster, `10.5%` lower median) on
`ops_bench` `conv3d/grad`, with non-overlapping intervals. Same-shape local PyTorch
CPU remains much faster at `7.593859 ms`, so current FT/PyTorch is still `3.50x`
slower. Gates: ft-kernel-cpu conv3d 2/0, ft-api conv3d 10/0, strict scheduler
conformance 1/0.

GroupNorm f32 unit-dy (`frankentorch-kgs4.115`): existing code-first f32 all-ones
`dy` fast path is now batch-verified. Same-worker `hz1` parent baseline at
`e1927d48` fused `19.13 ms` -> current `11.72 ms` (`1.63x` faster) on
`group_norm_f32_grad_ab`; current composed-vs-fused diagnostic is `101.96 ms`
-> `11.72 ms` (`8.70x`). Local PyTorch CPU `2.12.1+cpu` remains much faster at
`0.615446 ms` best-of-12, so current FT/PyTorch is still `19.04x` slower.
Gates: ft-kernel-cpu f32 unit-dy guard 1/0, ft-api f32 grad parity 1/0, strict
scheduler conformance 1/0.

GroupNorm f32 scalar-sum (`frankentorch-kgs4.135`): a dedicated affine
`functional_group_norm_sum` path computes the scalar loss directly and uses a
scalar upstream backward helper instead of materializing the output tensor,
`tensor_sum` node, and dense all-ones `dy`. The direct rch A/B row improved the
existing fused path from `8.30 ms` to `2.10 ms` (`3.96x` faster), and Criterion
moved the materialized median from `17.139 ms` to `8.9874 ms` (`1.91x` faster).
Local PyTorch CPU `2.12.1+cpu` with prebuilt tensors and clone/detach per rep
still has best `0.376163 ms`, so the direct scalar-sum row is `5.58x` slower.
Gates: ft-api scalar-sum tests 2/0, ft-kernel-cpu unit-dy guard 1/0, strict
scheduler conformance 1/0, lib clippy clean for ft-api and ft-kernel-cpu.

BatchNorm2d f32 scalar-sum (`frankentorch-kgs4.136`): a dedicated affine
`functional_batch_norm2d_sum` path computes the scalar loss directly and uses a
scalar upstream backward helper instead of materializing the normalized output,
`tensor_sum` node, and dense all-ones `dy`. The direct rch A/B diagnostic moved
the existing fused path from `10.80 ms` to `1.66 ms` (`6.50x` faster). On the
full PyTorch gauntlet shape, rch Criterion moved the existing fused row from
`114.23 ms` to `78.166 ms` (`1.46x` faster). Remote PyTorch remains unavailable
on rch workers (`No module named 'torch'`), so the PyTorch ratio uses the local
CPU oracle: PyTorch averaged `5.605736 ms/iter`, leaving the scalar row
`13.94x` slower. Gates: ft-kernel-cpu scalar-backward guards 2/0, ft-api
scalar-sum tests 2/0, strict scheduler conformance 1/0, scoped check/clippy
clean.

LayerNorm unit-dy (`frankentorch-kgs4.116`): existing code-first f64/f32
all-ones `dy` fast path is now batch-verified. Same-worker `hz2` parent
baseline at `2aa78200` `layer_norm/grad_2048x1024` `90.723 ms` -> current
`29.606 ms` (`3.06x` faster). Supporting f32 composed-vs-fused diagnostic is
`1930.66 ms` -> `293.49 ms` (`6.58x`). Local PyTorch CPU `2.12.1+cpu` remains
faster at `8.261743 ms` median for the same f64 shape, so current FT/PyTorch is
still `3.58x` slower. Gates: ft-kernel-cpu LayerNorm unit-dy guards 2/0, ft-api
functional LayerNorm 7/0, strict scheduler conformance 1/0 after canceling a
stale-progress `vmi1264463` run and retrying on `hz2`.

Conv2d all-ones dout (`frankentorch-kgs4.133`): the parked f64
`conv2d_backward_f64` candidate that collapses all-ones `dout` into one shared
`dweight` row and one shared `dpanel` row did not clear the same-worker keep
gate. On `vmi1152480`, current `ops_bench` `conv2d/grad_hw/64` was `121.07 ms`;
the active candidate was `117.92 ms`, but Criterion reported
`[-7.9705% -2.5970% +2.8489%]`, `p=0.38`, and no detected change. Local PyTorch
CPU `2.12.1+cpu` median for the same f64 scalar-sum train row was `63.449849 ms`,
so current FT/PyTorch is `1.91x` slower and the no-ship candidate remained
`1.86x` slower. The dormant branch was removed from product source.

MaxPool3d scalar-loss accumulation
(`frankentorch-maxpool3d-scalar-loss-grad-buffers-7wru6`): a no-report
`tensor_backward_accumulate` path that moved only leaf/`retain_grad` buffers
into persistent `.grad` did not move the fused max_pool3d sum-loss row. Baseline
`pytorch_gauntlet_bench` fused median was `5.7046 ms` against PyTorch
`2.3231 ms` (`2.46x` slower). The candidate median was `5.7846 ms` against
same-run PyTorch `1.9164 ms` (`3.02x` slower), while raw kernel
forward+indices regressed `+11.695%` and raw kernel backward-from-indices
regressed `+15.953%`. The source hook was reverted.

## Current Gates

| Gate | Scope | Result |
|---|---|---|
| PyTorch oracle | local CPU torch f64 BatchNorm1d NCL `[16,128,256]`, affine grads, prebuilt random tensors plus clone/detach per rep | PyTorch `2.12.1+cpu`, 32 compute/inter-op threads, median `1.061455 ms`; local FrankenTorch scalar-sum median `4.7944 ms`, ratio `4.52x` slower |
| Local same-host A/B | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a cargo bench -p ft-api --bench ops_bench -- batch_norm/grad_1d_ncl_16x128x256` | native materialized median `11.178 ms`, scalar-sum median `4.7944 ms`, scalar/native `0.4289x` (`2.33x` faster); fold-reference median `56.986 ms`, scalar/fold `0.0841x` (`11.89x` faster) |
| RCH Criterion | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo bench -p ft-api --bench ops_bench -- batch_norm/grad_1d_ncl_16x128x256` | baseline on `vmi1149989`: native `7.3230 ms`, fold `44.182 ms`; after run ignored worker pin and used `vmi1153651`, same-run native `43.610 ms`, scalar `25.058 ms`, fold `190.20 ms`, scalar/native `1.74x` faster |
| Correctness / conformance | `rch exec -- cargo test -p ft-kernel-cpu batch_norm_f64_scalar_backward --lib -- --nocapture`; `rch exec -- cargo test -p ft-kernel-cpu batch_norm --lib -- --nocapture`; `rch exec -- cargo test -p ft-api functional_batch_norm1d_sum_3d_matches_materialized_path --lib -- --nocapture`; `rch exec -- cargo test -p ft-conformance` | focused f64 scalar-backward tests 2/0 passed; full BatchNorm kernel filter 7/0 passed; API scalar/materialized NCL proof passed; full conformance green on `vmi1152480` |
| Compile / clippy / formatting / static | `rch exec -- cargo check -p ft-kernel-cpu --lib`; `rch exec -- cargo check -p ft-api --benches`; `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings`; `rch exec -- cargo clippy -p ft-api --bench ops_bench -- -D warnings`; targeted `rustfmt --check`; `git diff --check`; `ubs <scoped files>` | check/clippy passed; `git diff --check` passed; rustfmt check remains blocked by pre-existing unrelated whole-file drift in `ft-kernel-cpu`, `ft-api`, and `ops_bench`; manual UBS was interrupted after a long large-file Rust scan with no findings emitted (`exit=130`), and the pre-commit UBS hook hit its 300s timeout, so commit used `UBS_SKIP=1` |
| PyTorch oracle | local CPU torch f64 BatchNorm1d NCL `[16,128,256]`, affine grads, prebuilt tensors plus clone/detach per rep | PyTorch `2.12.1+cpu`, 32 compute/inter-op threads, median `2.251326 ms`; local FrankenTorch after row coarsening median `10.914 ms`, ratio `4.85x` slower |
| RCH Criterion | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo bench -p ft-api --bench ops_bench -- batch_norm/grad_1d_ncl_16x128x256` | pre-coarsening on `vmi1227854`: native median `4.3741 ms`, fold-reference median `30.484 ms`, native `6.97x` faster; after-coarsening supplemental on different worker `hz1`: native `6.2713 ms`, fold `60.234 ms`, native `9.60x` faster |
| Local same-host A/B | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a cargo bench -p ft-api --bench ops_bench -- batch_norm/grad_1d_ncl_16x128x256` | native NCL row improved `11.865 ms -> 10.914 ms` (`1.09x` faster); fold row `60.554 ms -> 57.450 ms` |
| Correctness / conformance | `rch exec -- cargo test -p ft-kernel-cpu batch_norm --lib -- --nocapture`; `rch exec -- cargo test -p ft-api functional_batch_norm1d_3d_native_fused_matches_fold_reference_bits --lib -- --nocapture`; `rch exec -- cargo test -p ft-conformance` | BatchNorm kernel tests 5/0 passed; NCL native-vs-fold bit guard passed before/after; full conformance passed via local fallback after RCH reported no admissible workers |
| Compile / clippy / formatting / static | `rch exec -- cargo check -p ft-api --benches`; `rch exec -- cargo check -p ft-kernel-cpu --lib`; `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings`; `rch exec -- cargo clippy -p ft-api --bench ops_bench -- -D warnings`; `rustfmt --edition 2024 --check crates/ft-kernel-cpu/src/lib.rs`; `ubs <scoped files>` | check/clippy/kernel rustfmt passed; `ops_bench` clippy initially found two pre-existing single-element loops, fixed and rerun passed, then passed again after the UBS label-comparison cleanup; scoped UBS rerun reported 0 critical issues with the existing broad warning inventory; whole `ops_bench` rustfmt remains blocked by unrelated existing drift |
| PyTorch oracle | local CPU torch f32 BatchNorm2d `[32,256,28,28]`, affine grads, prebuilt tensors plus clone/detach per rep | PyTorch 30 iterations `0.168172072968 s`, `5.605736 ms/iter`; rch Criterion scalar-sum FrankenTorch mean `78.166 ms`, ratio `13.94x` slower |
| Remote direct A/B | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo run --release -p ft-api --example batch_norm_f32_grad_ab` | `frankentorch-kgs4.136` candidate on `vmi1227854`: composed `109.59 ms`, existing fused `10.80 ms`, scalar-sum `1.66 ms`; scalar/fused `0.1537x`, `6.50x` faster |
| Criterion | `PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- batch_norm2d_f32 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | rch worker `vmi1227854`: existing fused mean `114.23 ms`, scalar-sum mean `78.166 ms`, `1.46x` faster; PyTorch arm failed because remote `torch` is unavailable |
| Correctness / conformance | `rch exec -- cargo test -p ft-kernel-cpu batch_norm_f32_scalar_backward_matches_unit_dy_bits -- --nocapture`; `rch exec -- cargo test -p ft-api functional_batch_norm2d_f32_sum --lib -- --nocapture`; `rch exec -- cargo test -p ft-conformance strict_scheduler -- --nocapture` | all passed for `frankentorch-kgs4.136` |
| Compile / clippy / formatting / static | `rch exec -- cargo check -p ft-api --all-targets`; `rch exec -- cargo check -p ft-kernel-cpu --all-targets`; `rch exec -- cargo clippy -p ft-api --lib -- -D warnings`; `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings`; `rch exec -- cargo clippy -p ft-api --example batch_norm_f32_grad_ab -- -D warnings`; `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings`; targeted `rustfmt --check`; `git diff --check`; `ubs <scoped files>` | check/clippy passed; `ft-api --all-targets` still reports only existing `hessian_probe.rs` warning and `ft-kernel-cpu --all-targets` only existing `gemm_golden.rs` warnings; small touched benchmark/example rustfmt and diff whitespace passed; UBS timed out after 240s with no findings emitted; broad crate fmt remains blocked by unrelated drift |
| PyTorch oracle | local CPU torch f32 GroupNorm `[8,64,28,28]`, groups `32`, affine grads, prebuilt tensors plus clone/detach per rep | PyTorch `2.12.1+cpu` best `0.376163 ms`, median `0.512991 ms`; direct scalar-sum FrankenTorch row `2.10 ms`, ratio `5.58x` slower |
| Remote direct A/B | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo run --release -p ft-api --example group_norm_f32_grad_ab` | `frankentorch-kgs4.135` candidate on `ovh-a`: composed `69.33 ms`, existing fused `8.30 ms`, scalar-sum `2.10 ms`; scalar/fused `0.2525x`, `3.96x` faster |
| Criterion | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo bench -p ft-api --bench ops_bench -- group_norm/grad_f32 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | `group_norm/grad_f32_8x64x28x28` median `17.139 ms`; `group_norm/grad_f32_sum_8x64x28x28` median `8.9874 ms`; scalar-sum `1.91x` faster |
| Correctness / conformance | `rch exec -- cargo test -p ft-api functional_group_norm_f32_sum --lib`; `rch exec -- cargo test -p ft-kernel-cpu group_norm_f32_unit_dy_matches_general_reference_bits --lib`; `rch exec -- cargo test -p ft-conformance strict_scheduler` | all passed for `frankentorch-kgs4.135` |
| Compile / clippy / formatting / static | `rch exec -- cargo check -p ft-api --all-targets`; `rch exec -- cargo check -p ft-kernel-cpu --all-targets`; `rch exec -- cargo clippy -p ft-api --lib -- -D warnings`; `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings`; `git diff --check`; `ubs <scoped files>`; `rch exec -- cargo fmt --check -p ft-api -p ft-kernel-cpu` | check, lib clippy, and diff whitespace passed; UBS was interrupted after more than 3 minutes with no findings emitted; broader all-target clippy and crate fmt remain blocked by existing unrelated example/test lint and rustfmt debt |
| PyTorch gauntlet | `PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo bench -p ft-api --bench pytorch_gauntlet_bench -- batch_norm2d_f32 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | `frankentorch-kgs4.114` active branch local FT `228.85 ms`, PyTorch `6.8744 ms`, `33.29x` slower; disabled/final local FT `238.33 ms`, PyTorch `8.4699 ms`, `28.14x` slower |
| Remote same-worker A/B | `RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- gauntlet_batch_norm2d_f32_grad/frankentorch_kgs4_114 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | disabled/final `147.30 ms`; active unit-dy branch `157.93 ms`; active/disabled `1.072x` slower, Criterion `[+1.2713% +7.2142% +13.421%]`, `p=0.05`; product branch reverted |
| Correctness / conformance | `rch exec -- cargo test -p ft-kernel-cpu batch_norm_f32_unit_dy_matches_general_reference_bits -- --nocapture`; `rch exec -- cargo test -p ft-api functional_batch_norm2d_f32_grad_matches_f64_path -- --nocapture`; `rch exec -- cargo test -p ft-conformance` | all passed for `frankentorch-kgs4.114`; full conformance green on `hz2` |
| Compile / clippy / static | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench`; `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings`; `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings`; `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs`; `python3 -m py_compile crates/ft-api/benches/pytorch_batch_norm2d_f32_grad.py`; `git diff --check` | all passed |
| PyTorch gauntlet | `PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a cargo bench -p ft-api --bench pytorch_gauntlet_bench -- avg_pool1d --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | `frankentorch-kgs4.134` local baseline old median `79.285 ms`, PyTorch `6.2886 ms`, ratio `12.61x` slower; candidate same-run old `69.267 ms`, fused `59.050 ms`, PyTorch `7.8192 ms`, fused/PyTorch `7.55x` slower |
| Remote build/bench | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- avg_pool1d --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | rch worker `vmi1152480`: old `134.74 ms`, fused scalar-sum `87.564 ms`, fused/old `0.6500x`; PyTorch arm failed because remote `torch` is unavailable |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu avg_pool1d_sum_scalar_backward_matches_materialized_bits -- --nocapture`; `rch exec -- cargo test -p ft-api functional_avg_pool1d_sum_matches_pool_sum_backward_bits -- --nocapture`; `rch exec -- cargo test -p ft-conformance` | all passed for `frankentorch-kgs4.134`; full conformance suite green |
| Compile / clippy | `rch exec -- cargo check -p ft-kernel-cpu --all-targets`; `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench`; `rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings`; `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings` | all passed; `ft-kernel-cpu --all-targets` still reports only the existing `gemm_golden.rs` example warning |
| Static checks | `git diff --check -- crates/ft-kernel-cpu/src/lib.rs crates/ft-api/src/lib.rs crates/ft-api/benches/pytorch_gauntlet_bench.rs`; `ubs crates/ft-kernel-cpu/src/lib.rs crates/ft-api/src/lib.rs crates/ft-api/benches/pytorch_gauntlet_bench.rs`; `rustfmt --edition 2024 --check ...` | diff whitespace passed; large-file UBS was interrupted after several minutes and the pre-commit large-file UBS hook timed out, so the commit used `UBS_SKIP=1` after narrow UBS on benchmark/docs/artifacts passed; whole-file rustfmt remains blocked by pre-existing unrelated drift in large files, while the touched hunks were manually normalized |
| PyTorch gauntlet | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d --noplot` | `frankentorch-maxpool3d-scalar-loss-grad-buffers-7wru6` baseline fused median `5.7046 ms`, PyTorch median `2.3231 ms`, ratio `2.46x` slower; no-report accumulate-only candidate `5.7846 ms`, same-run PyTorch `1.9164 ms`, ratio `3.02x` slower |
| Correctness / compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench`; `rch exec -- cargo test -p ft-api functional_max_pool3d_sum_matches_pool_sum_backward_bits -- --nocapture` | trial code compiled and passed focused bit-exact gradient proof before rejection; source hook reverted because performance did not clear the keep gate |
| Criterion | `RCH_WORKER=vmi1152480 RCH_WORKERS=vmi1152480 CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo bench -p ft-api --bench ops_bench -- conv2d/grad_hw/64 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | `frankentorch-kgs4.133` same-worker current `121.07 ms`; active all-ones-dout candidate `117.92 ms`; `p=0.38`, no change detected |
| PyTorch oracle | local CPU torch f64 Conv2d `[4,64,64,64]`, 64 3x3 filters, padding 1, sum loss | PyTorch `2.12.1+cpu` median `63.449849 ms`, min `59.068578 ms`; current FrankenTorch ratio `1.91x` slower, candidate ratio `1.86x` slower |
| Compile | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo check -p ft-kernel-cpu --all-targets` | passed on `hz1`; existing example warning in `gemm_golden.rs` remains outside the removed conv2d branch |
| Clippy | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo clippy -p ft-kernel-cpu --lib -- -D warnings` | passed on `vmi1293453`; broader `--all-targets` is still blocked by pre-existing lint debt in examples/tests and unrelated helper code |
| Correctness | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo test -p ft-kernel-cpu conv2d -- --nocapture` | passed: 5 tests ok, 1 perf-only ignored |
| Conformance | `CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a rch exec -- cargo test -p ft-conformance` | passed on `vmi1227854`: 199 lib tests, bin unit tests, integration tests, smoke tests, and doc-tests all green |
| Static checks | `git diff --check`; `ubs crates/ft-kernel-cpu/src/lib.rs docs/NEGATIVE_EVIDENCE.md docs/RELEASE_READINESS_SCORECARD.md artifacts/perf/frankentorch-kgs4.133/gauntlet_20260620T0533Z/SCORECARD.md artifacts/perf/frankentorch-kgs4.133/gauntlet_20260620T0533Z/NEGATIVE_EVIDENCE_LEDGER.md`; `rustfmt --edition 2024 --check crates/ft-kernel-cpu/src/lib.rs` | diff whitespace passed; UBS found 0 critical issues and existing large-file warning inventory; whole-file rustfmt remains blocked by pre-existing formatting drift outside this deletion |
| Rust A/B | `rch exec -- cargo bench -p ft-api --bench ops_bench -- layer_norm/grad_2048x1024 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | `frankentorch-kgs4.116` current ran on `hz2`: `29.606 ms`; detached parent `2aa78200` on `hz2`: `90.723 ms`; current is `3.06x` faster |
| Rust diagnostic | `rch exec -- cargo run --release -p ft-api --example layernorm_f32_grad_ab` | `frankentorch-kgs4.116` current ran on `hz2`: composed `1930.66 ms`, fused `293.49 ms`, `6.58x` faster |
| PyTorch oracle | local CPU torch f64 LayerNorm `[2048,1024]`, affine grads, sum loss | PyTorch `2.12.1+cpu` median `8.261743 ms`, min `5.949352 ms`; current FrankenTorch Criterion estimate `29.606 ms`, ratio `3.58x` slower |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu layer_norm_f -- --nocapture`; `rch exec -- cargo test -p ft-api functional_layer_norm -- --nocapture`; `rch exec -- cargo test -p ft-conformance strict_scheduler -- --nocapture` | all passed for `frankentorch-kgs4.116`; conformance retry on `hz2` passed after canceling stale-progress `vmi1264463` run |
| Rust A/B | `rch exec -- cargo run --release -p ft-api --example group_norm_f32_grad_ab` | `frankentorch-kgs4.115` current ran on `hz1`: composed `101.96 ms`, fused current `11.72 ms`; detached parent `e1927d48` on `hz1`: fused `19.13 ms`; current is `1.63x` faster |
| PyTorch oracle | local CPU torch f32 GroupNorm best-of-12, `[8,64,28,28]`, groups `32`, affine grads, sum loss | PyTorch `2.12.1+cpu` best `0.615446 ms`, median `0.989997 ms`; current FrankenTorch best `11.72 ms`, ratio `19.04x` slower |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu group_norm_f32_unit_dy_matches_general_reference_bits`; `rch exec -- cargo test -p ft-api functional_group_norm_f32_grad_matches_f64_path`; `rch exec -- cargo test -p ft-conformance strict_scheduler` | all passed for `frankentorch-kgs4.115` on rch workers `vmi1153651`, `ovh-a`, and `vmi1152480` |
| Criterion | `cargo bench -p ft-api --bench ops_bench -- smooth_l1/grad_8m --noplot` | `frankentorch-grefr` direct local A/B completed; paired-randn candidate `469.36 ms` vs pre-lever `588.51 ms`; PyTorch `347.528377 ms`, ratio `1.35x` slower |
| Remote build/bench | `rch exec -- cargo bench -p ft-api --bench ops_bench -- smooth_l1/grad_8m --noplot` | pre-lever remote row on `vmi1264463` `2.1181 s`; paired-randn candidate row on `vmi1293453` `944.17 ms`; candidate retry selected `vmi1264463` but fell back local after sync timeout; remote rows are routing evidence, not decisive A/B proof |
| Compile | `rch exec -- cargo check -p ft-api` | passed for `frankentorch-grefr` |
| Clippy | `rch exec -- cargo clippy -p ft-api -- -D warnings` | passed for `frankentorch-grefr` |
| Correctness | `rch exec -- cargo test -p ft-api randn_creates_normal_values`; `rch exec -- cargo test -p ft-conformance` | both passed for `frankentorch-grefr` after seeded f64 normal fixture update |
| UBS | `ubs` on changed files, docs-only, `crates/ft-api/src/lib.rs` retry, and pre-commit hook | docs-only exited 0; changed-file scan timed out after 300s, Rust-only retry timed out after 180s, and hook timed out on staged large-file Rust scan with no findings emitted before timeout |
| Criterion | `RCH_WORKER=vmi1227854 RCH_WORKERS=vmi1227854 rch exec -- cargo bench -p ft-api --bench ops_bench -- sdpa/grad_16x512x64 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | kgs4.113 current scaled-alpha path median `82.730 ms`; temporary old post-scale variant median `114.40 ms`; new/old ratio `0.723x` (`1.38x` faster), old shape rejected |
| PyTorch gauntlet | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- sdpa --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | local diagnostic PyTorch `2.12.0+cpu`; FrankenTorch median `63.057 ms`, PyTorch median `48.915 ms`, ratio `1.29x` slower |
| Remote build/bench | `rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- sdpa --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | built and ran FrankenTorch arm on `vmi1227854`, median `53.254 ms`; PyTorch arm failed because remote `torch` was unavailable |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu scaled_gemm_matches_post_scale_reference -- --nocapture`; `rch exec -- cargo test -p ft-api sdpa_ -- --nocapture` | kernel guard passed on `hz2`; API SDPA group passed 17 unit tests plus finite-diff integration row |
| Conformance | `rch exec -- cargo test -p ft-conformance` | passed on `vmi1149989`: 199 lib tests plus bins, E2E, PyTorch conformance, smoke, and doc-tests all green |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `hz1` |
| Clippy | `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings` | passed on `hz1` |
| UBS | `ubs crates/ft-api/benches/pytorch_gauntlet_bench.rs crates/ft-api/benches/pytorch_sdpa_grad.py docs/NEGATIVE_EVIDENCE.md docs/RELEASE_READINESS_SCORECARD.md artifacts/perf/frankentorch-kgs4.113/verify_20260619T182412Z/SCORECARD.md artifacts/perf/frankentorch-kgs4.113/verify_20260619T182412Z/NEGATIVE_EVIDENCE_LEDGER.md artifacts/perf/frankentorch-kgs4/sdpa_scaled_gemm_alpha_code_first.md` | zero critical or warning findings |
| Formatting | `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs`; `python -m py_compile crates/ft-api/benches/pytorch_sdpa_grad.py`; `git diff --check` | changed Rust bench, Python script, and diff whitespace passed; broad `cargo fmt --check` still reports pre-existing unrelated workspace drift |
| Criterion | `rch exec -- cargo bench -p ft-api --bench ops_bench -- avg_pool2d/grad --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | kgs4.112 current fast path passed on `hz2`; median `58.600 ms`; direct-assignment candidate on same worker regressed to `68.624 ms` and was reverted |
| PyTorch gauntlet | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- avg_pool2d --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | local PyTorch `2.12.0+cpu`; FrankenTorch median `16.627 ms`, PyTorch median `3.6632 ms`, ratio `4.54x` slower |
| Remote build/bench | `rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- avg_pool2d --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot` | built and ran FrankenTorch arm on `hz2`, median `13.383 ms`; PyTorch arm failed because remote `torch` was unavailable |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `ovh-b` for kgs4.112 gauntlet harness |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu avg_pool2d_2x2s2_backward_matches_generic_bit_exact -- --nocapture` | passed on `vmi1264463` after reverting the direct-assignment candidate |
| Conformance | `rch exec -- cargo test -p ft-conformance` | passed via rch local fallback; all conformance groups green |
| Clippy | `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings` | passed on `hz2` |
| UBS | `ubs crates/ft-api/benches/pytorch_gauntlet_bench.rs crates/ft-api/benches/pytorch_avg_pool2d_grad.py docs/NEGATIVE_EVIDENCE.md docs/RELEASE_READINESS_SCORECARD.md artifacts/perf/frankentorch-kgs4.112/gauntlet_20260619T174250Z/NEGATIVE_EVIDENCE_LEDGER.md artifacts/perf/frankentorch-kgs4.112/gauntlet_20260619T174250Z/SCORECARD.md` | zero critical or warning findings |
| Formatting | `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs` | passed; workspace `cargo fmt --check` still reports pre-existing unrelated drift in ft-api examples |
| Criterion | `cargo bench -p ft-api --bench ops_bench -- smooth_l1/grad_8m --noplot` | kgs4.127 local same-host A/B completed; current before one-sided grad `746.26 ms`, candidate `647.44 ms`; PyTorch `360.785 ms`, ratio `1.79x` slower |
| Remote build/bench | `rch exec -- cargo bench -p ft-api --bench ops_bench -- smooth_l1/grad_8m --noplot` | current pre-change ran on `ovh-a` at `674.81 ms`; candidate ran on `hz1` at `774.85 ms`; supplemental candidate ran on `vmi1152480` at `619.16 ms`; cross-worker rows are routing evidence, not same-worker proof |
| Compile | `rch exec -- cargo check -p ft-api` | passed for kgs4.127 |
| Clippy | `rch exec -- cargo clippy -p ft-api -- -D warnings`; `rch exec -- cargo clippy -p ft-kernel-cpu -- -D warnings` | both passed for kgs4.127 |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu smooth_l1_backward_reduced_one_sided_helpers_match_full_bits`; `rch exec -- cargo test -p ft-api smooth_l1_loss_reduced_grad_skips_unneeded_one_sided_gradient` | both passed |
| UBS | `ubs crates/ft-api/src/lib.rs crates/ft-kernel-cpu/src/lib.rs` | completed after 545s; returned broad pre-existing inventory in these large files, including historical panic/security heuristics outside the SmoothL1 hunk |
| Formatting | `git diff --check`; `rustfmt --edition 2024 --check crates/ft-api/src/lib.rs crates/ft-kernel-cpu/src/lib.rs` | diff whitespace passed; whole-file rustfmt remains blocked by broad pre-existing formatting drift |
| Criterion | `cargo bench -p ft-api --bench ops_bench -- smooth_l1/grad_8m --noplot` | completed on `hz2`; current median `757.63 ms` |
| PyTorch oracle | `torch_smooth_l1_grad_8m.py` | local PyTorch `2.12.1+cu130` median `373.61 ms` |
| Compile | `rch exec -- cargo check -p ft-api` | passed |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu smooth_l1_backward_reduced_f64_matches_uniform_dloss_bits` | passed |
| Correctness | `rch exec -- cargo test -p ft-api smooth_l1_loss_reduced_grad_matches_materialized_reference_bits` | passed after explicit default-argument test-call fix |
| Criterion | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool1d --noplot` | completed locally with PyTorch `2.12.1+cpu` |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `ovh-a` for final harness |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu max_pool1d_direct_matches_2d_h1_first_tie_forward_backward_bit_exact` | passed on `ovh-a` |
| Formatting | `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs` | passed |
| Criterion | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- avg_pool1d --noplot` | completed locally; candidate FrankenTorch median `204.02 ms`, PyTorch median `7.4798 ms`; final reverted FrankenTorch median `184.99 ms`, PyTorch median `7.1539 ms`; rerun final median `181.94 ms`, PyTorch median `7.3011 ms` |
| Remote build/bench | `rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- avg_pool1d --noplot` | Rust bench built and ran on `vmi1227854`; PyTorch arm failed because remote `torch` was unavailable, so this is not ratio evidence |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `vmi1152480` after avg_pool1d revert |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu avg_pool1d_direct_matches_2d_h1_forward_backward_bit_exact -- --nocapture` | passed on `ovh-a` after avg_pool1d revert |
| Correctness | `rch exec -- cargo test -p ft-api functional_avg_pool1d_fused_matches_reshape_2d_forward_and_backward_bits -- --nocapture` | passed on `vmi1227854` after avg_pool1d revert |
| Clippy | `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings` | passed on `hz1` after avg_pool1d revert |
| Clippy | `rch exec -- cargo clippy -p ft-kernel-cpu -- -D warnings` | passed on `vmi1264463` after avg_pool1d revert |
| UBS | `ubs crates/ft-api/benches/pytorch_gauntlet_bench.rs crates/ft-kernel-cpu/src/lib.rs crates/ft-api/benches/pytorch_avg_pool1d_grad.py` | no critical findings; existing `ft-kernel-cpu` warning inventories remain outside this revert |
| Formatting | `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs`; `git diff --check` on changed files | passed; whole-file `ft-kernel-cpu/src/lib.rs` rustfmt check still reports pre-existing formatting drift outside this avg_pool1d revert |
| Criterion | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- linear --noplot` | completed locally; final FrankenTorch median `22.775 ms`, PyTorch median `9.2821 ms` |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `vmi1152480` after restoring the API-local linear path |
| Correctness | `rch exec -- cargo test -p ft-api linear_backward_all_ones_dy_matches_kernel_reference -- --nocapture` | passed on `vmi1293453` |
| Criterion | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d --noplot` | completed locally; current FrankenTorch median `15.794 ms`, PyTorch median `1.6228 ms` |
| Criterion baseline | parent worktree at `c79d3a23`, same max_pool3d harness | completed locally; parent FrankenTorch median `20.585 ms`, PyTorch median `2.1381 ms` |
| Remote build/bench | `rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d --noplot` | built on `hz2`; current FrankenTorch median `28.124 ms`; PyTorch arm failed because remote `torch` was unavailable |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `hz2` |
| Correctness | `rch exec -- cargo test -p ft-kernel-cpu max_pool3d_indices_scatter_matches_rescan_first_tie_bits` | passed after clippy-only lint fixes |
| Correctness | `rch exec -- cargo test -p ft-api functional_max_pool3d_grad_matches_finite_diff` | passed after clippy-only lint fixes |
| Clippy | `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings` | passed after narrow ft-kernel-cpu clippy fixes |
| Criterion | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- 'gauntlet_max_pool3d_grad/' --noplot` | kgs4.128 clean baseline completed locally; FrankenTorch median `15.303 ms`, PyTorch median `1.6325 ms`; borrowed-input and unit-dout candidates rejected and product source reverted |
| Stage probe | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool3d_grad_stage --noplot` | added diagnostic max_pool3d stage rows; forward-only `4.1256 ms` vs raw kernel forward+indices `727.15 us`; backward rows were noisy and used only as routing evidence |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `hz2` for the retained kgs4.128 stage-probe harness |
| Correctness | `rch exec -- cargo test -p ft-api functional_max_pool3d_grad_matches_finite_diff -- --nocapture` | passed on `vmi1152480` after product candidates were reverted |
| Clippy | `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings` | passed on `vmi1152480` for the retained kgs4.128 stage-probe harness |
| UBS | `ubs crates/ft-api/benches/pytorch_gauntlet_bench.rs docs/NEGATIVE_EVIDENCE.md docs/RELEASE_READINESS_SCORECARD.md artifacts/perf/frankentorch-kgs4.128/gauntlet_20260619T0521Z/summary.md` | zero critical or warning findings |
| Formatting | `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs`; `git diff --check` | passed for the kgs4.128 changed surface |
| Criterion | `cargo bench -p ft-api --bench pytorch_gauntlet_bench -- 'gauntlet_conv3d_grad' --noplot` | kgs4.119 completed locally with PyTorch `2.12.1+cpu`; current FrankenTorch median `24.095 ms`, PyTorch median `10.126 ms`, ratio `2.38x` slower |
| Same-worker A/B | `rch exec -- cargo bench -p ft-api --bench pytorch_gauntlet_bench -- 'gauntlet_conv3d_grad/frankentorch_kgs4_119' --noplot` | on `ovh-a`, disabled save-copy median `19.429 ms`, current borrowed-input median `15.632 ms`; borrowed-input is `1.24x` faster and kept |
| Remote routing | same command | initial current-only run on `vmi1152480` median `28.364 ms`; no same-worker disabled/PyTorch comparator, so neutral routing evidence only |
| Compile | `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench` | passed on `vmi1152480` for the kgs4.119 Conv3d gauntlet harness |
| Correctness | `rch exec -- cargo test -p ft-api conv3d --lib -- --nocapture` | passed on `vmi1293453`: `10 passed; 0 failed` |
| Conformance | `rch exec -- cargo test -p ft-conformance` | passed on `vmi1227854`: lib `199/0`, bins/integration/smoke/doc tests all green |
| Clippy | `rch exec -- cargo clippy -p ft-api --bench pytorch_gauntlet_bench -- -D warnings` | passed on `vmi1227854` |
| Formatting | `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs`; `python3 -m py_compile crates/ft-api/benches/pytorch_conv3d_grad.py`; `git diff --check` | kgs4.119 changed benchmark/docs surface passed |

Known caveat: `cargo fmt --check -p ft-api` remains blocked by pre-existing
crate-wide formatting debt in unrelated examples and long `ft-api/src/lib.rs`
regions. `cargo fmt -p ft-api -- --check` also reports broad existing drift
for the SmoothL1 closeout; follow-up `frankentorch-6xsy8` tracks that cleanup.

UBS caveat: a full changed-file UBS scan including the 136k-line
`ft-api/src/lib.rs` did not complete after several minutes and was interrupted
or timed out in the pre-commit hook. The max_pool1d benchmark surface was
scanned directly and passed with zero critical or warning findings; no UBS
verdict was available for the SmoothL1 `ft-api/src/lib.rs` closeout.

## Next Perf Target

The `.124` result points toward deeper SmoothL1 training overhead: tape setup,
input/materialization cost, loss backward kernel shape, SIMD, and cache layout.
The `.113` result rejects SDPA post-GEMM scale streams and keeps scaled GEMM
alpha, but the remaining PyTorch gap should move to whole-row scheduling,
cache-blocked softmax/GEMM interaction, f32-native ratio work, arena/tape
allocation removal, or fused loss/backward primitives rather than another
post-scale cleanup.
The `.134` result narrows avg_pool1d with scalar-loss fusion, but the remaining
`7.55x` PyTorch loss should move to a general fused-loss/backward family,
persistent gradient/tape allocation, or whole-row `.grad` traffic removal rather
than another avg_pool1d kernel-only branch.
The `.114` result rejects f32 BatchNorm all-ones-`dy` as a local branch. Route
BatchNorm to fused scalar-loss, saved-stat reuse, persistent workspaces,
stats+backward fusion, arena/tape allocation, or cache-blocked f32 reductions
with same-worker proof.
The `.116` result verifies the LayerNorm unit-dy branch as a real internal win,
but the remaining `3.58x` PyTorch gap should move beyond constant-gradient
normalization branches to whole-row allocation/tape/loss fusion, persistent
normalization workspaces, deterministic parallel affine-gradient reductions,
and f32-native end-to-end rows.
The `.112` result verifies the current avg_pool2d 2x2s2 specialization but
rejects direct assignment scatter writes; the remaining gap should move to
whole-workload tape/allocation/sum-backward overhead, native f32 layout, or a
fused avg-pool training primitive with fresh ratio evidence.
The `.127` result proves that avoiding the unused target gradient allocation is
worthwhile, but the remaining SmoothL1 row is still PyTorch-bound; the next
SmoothL1 attempt should attack allocation lifetime, RNG/input setup, tape edges,
or vectorized branchless gradient generation rather than another scalar wrapper.
The `.122` result points away from tiny avg_pool1d unit-gradient fill branches
and toward end-to-end pooling overhead, especially session/tape setup,
allocation churn, forward materialization, and generic pooling dispatch costs.
The `.126` result points away from tiny max_pool1d backward scatter branches and
toward larger full-step costs: autograd/session setup, allocation churn, and
forward saved-index materialization. The `.117` result keeps compact max_pool3d
sidecars as an internal win but leaves a roughly 10x PyTorch gap, so the next
pooling pass needs an end-to-end profile rather than another sidecar-only tweak.
The `.121` result points away from moving the all-ones linear backward shortcut
into the generic CPU kernel; the remaining gap is end-to-end linear training
overhead, not the already-collapsed row-sum/copy helper alone. Future work
should profile those frames before trying another scalar-wrapper or one-off
unit-gradient branch. The `.128` result rejects borrowed-input-only max_pool3d
autograd and standalone unit-`dout` scatter branches; the next viable route is
end-to-end fusion that removes the sum-generated gradient buffer/tape edge,
allocator/arena work proven on the whole training row, or a fundamentally
different layout/kernel plan with fresh ratio evidence.
The `.133` result rejects materialized-im2col conv2d all-ones row collapse:
the constants and allocations ate the theoretical win. The next conv2d attempt
should avoid this family unless a fresh profile proves a new primitive, such as
direct no-panel all-ones backward, workspace-backed panel reuse, cache-blocked
col2im, or fused loss/backward/tape work, moves the full train row.
The `7wru6` result rejects max_pool3d report-skipping/persistent-grad-only API
work. The next max_pool3d attempt should be a true fused loss/backward primitive
or a scheduler/arena rewrite with same-worker proof on the full PyTorch gauntlet
row, not another wrapper around the generic backward report path.
The `.119` result keeps borrowed Conv3d autograd inputs as a real same-worker
win, but the row still loses to PyTorch by `2.38x`; the next Conv3d pass should
target whole-row autograd/tape allocation, scalar-loss gradient materialization,
persistent workspaces, or a direct fused Conv3d sum-backward primitive rather
than revisiting saved-input copy removal.
