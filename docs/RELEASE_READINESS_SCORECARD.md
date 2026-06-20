# FrankenTorch Release-Readiness Scorecard

Updated: 2026-06-19

## Performance Gauntlet

| Bead | Workload | Result vs PyTorch | Before/after verdict | Release action |
|---|---:|---:|---:|---|
| `frankentorch-kgs4.113` | SDPA f64 train step `[16,512,64]` | `1.29x` slower | internal keep; same-worker rch `114.40 ms` old post-scale -> `82.730 ms` scaled alpha | kept; route remaining gap to SDPA scheduling/allocation/fusion |
| `frankentorch-kgs4.112` | avg_pool2d f64 train step `[8,64,64,64]` | `4.54x` slower | existing 2x2s2 fast path verified; direct-assignment candidate `58.600 ms` -> `68.624 ms` rejected | keep gauntlet harness/evidence; product source unchanged |
| `frankentorch-kgs4.117` | max_pool3d f64 train step `[2,32,16,32,32]` | `9.73x` slower | internal keep; `20.585 ms` -> `15.794 ms`; remote PyTorch arm unavailable on `hz2` | kept; profile deeper end-to-end gap |
| `frankentorch-kgs4.121` | linear f64 train step `[32,512] -> 2048` | `2.45x` slower | API-local internal keep; `29.606 ms` -> `22.775 ms`; kernel move `26.459 ms` rejected | kept API helper; reverted kernel move |
| `frankentorch-kgs4.122` | avg_pool1d f64 train step `[8,64,8192]` | `25.86x` slower; final rerun `24.92x` slower | no gain; candidate median `204.02 ms` vs fast-path-disabled `179.91 ms` | reverted |
| `frankentorch-kgs4.124` | SmoothL1 f64 mean-loss backward, 8M elems | `1.99x` slower | internal keep; `963.16 ms` -> `757.63 ms` on `hz2` | kept; follow-up `frankentorch-kgs4.127` |
| `frankentorch-kgs4.126` | max_pool1d f64 train step `[8,64,8192]` | `12.31x` slower | no gain; candidate median `184.41 ms` vs parent `178.47 ms` | reverted |
| `frankentorch-kgs4.127` | SmoothL1 f64 one-sided input grad, 8M elems | `1.79x` slower | internal keep; same-host local `746.26 ms` -> `647.44 ms` | kept; route remaining gap to tape/allocation/SIMD |
| `frankentorch-kgs4.128` | max_pool3d f64 train step `[2,32,16,32,32]` | `9.38x` slower clean baseline | no gain; borrowed-input median `22.764 ms`, unit-dout median `16.160 ms`, sequential unit-dout median `22.465 ms` | reverted product candidates; keep stage probe |
| `frankentorch-grefr` | SmoothL1 f64 mean-loss backward, 8M elems | `1.35x` slower | internal keep; direct local `588.51 ms` -> `469.36 ms`; beta=1 derivative branch rejected | kept paired-randn fill; route remaining gap to tape/allocation/loss-kernel |

Measured-discipline score: `10/10` for the gauntlet lanes. PyTorch head-to-head
score: `0W / 10L / 0N`. Correctness guards are green and the SDPA, MaxPool3d, Linear,
and SmoothL1 levers include real internal speedups, but no measured workload is
performance-dominant against PyTorch yet.

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

## Current Gates

| Gate | Scope | Result |
|---|---|---|
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
