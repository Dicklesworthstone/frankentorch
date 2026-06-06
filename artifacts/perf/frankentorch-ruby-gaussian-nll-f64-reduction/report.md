# gaussian_nll f64 no-grad reduction map-reduce

Bead: `frankentorch-7f1m`
Original baseline commit: `4aafb31ddf6564f9c758f43508d1b8dc45b5c62b`
Final rebased parent: `452f41c0a8c122233d492bae2f5116ed798baba4`
Optimized candidate: removed after final gate failed
Agent: `RubyLotus`

## Target

`br ready --json --no-auto-import` had no ready perf beads; visible in-progress perf lanes were owned by other agents. I selected the disjoint existing Criterion target `gaussian_nll/nograd_8m`, which profiles same-shape f64 no-grad `mean`. The existing f64 fast path already fused the per-element Gaussian NLL expression, but still materialized the full 8M-element loss vector before `tensor_mean` or `tensor_sum`.

Candidate primitive tested: fused pairwise map-reduce / materialization removal. This computed each Gaussian NLL element at reduction leaves and preserved the existing midpoint pairwise reduction tree instead of allocating and then reducing a loss tensor.

## Benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-api --bench ops_bench -- gaussian_nll/nograd_8m --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Same worker: `ts1`

| Run | Criterion interval |
| --- | --- |
| Baseline `4aafb31d` | `[156.14 ms 166.56 ms 181.34 ms]` |
| Candidate on `4aafb31d` | `[138.22 ms 140.30 ms 142.35 ms]` |
| Rebased parent `0d2bb62b` | `[209.66 ms 215.46 ms 221.58 ms]` |
| Rebased optimized `21f4cfbc` | `[139.06 ms 140.53 ms 142.03 ms]` |
| Final parent `452f41c0` | `[146.75 ms 148.90 ms 151.20 ms]` |
| Final optimized candidate | `[147.25 ms 152.30 ms 158.25 ms]` |

Final current-parent median ratio: `148.90 / 152.30 = 0.98x`; the candidate regressed after rebasing onto `452f41c0`.
Earlier in-flight runs showed provisional wins (`166.56 / 140.30 = 1.19x`, then `215.46 / 140.53 = 1.53x`), but those were superseded by the final same-worker parent/head gate.

Score: below gate because the final current-parent run regressed. Verdict: reject and keep no source changes.

## Isomorphism Proof

Candidate ordering and tie-breaking:
The optimized route is only for same-shape f64 no-grad `mean` and `sum`. The `none` reduction keeps the old per-element tensor path, so output element ordering is unchanged there. Scalar reductions have no tie-breaking surface.

Candidate floating point:
The per-element expression is factored into `gaussian_nll_value_f64`, matching `gaussian_nll_forward_f64`: `0.5 * (var.ln() + d*d/var + c)`. The reducer splits at the same `mid = len / 2` boundaries, uses the same 128-element serial leaves and the same parallel threshold/tree shape as `sum_tensor_contiguous_f64`, and combines every node as `left + right`. `mean` divides the resulting sum by the same `n as f64`. Focused tests assert `to_bits()` equality for sum and mean against the materialized-vector path, for `full=false` and `full=true`.

RNG:
No RNG is introduced. The benchmark creates tensors before `b.iter`; the optimized path only reads tensor values and computes a deterministic scalar.

Autograd:
The fast path is gated on input, target, and var all not requiring gradients. Any gradient-bearing input still uses the existing custom autograd path. The existing Gaussian NLL finite-difference and gradient propagation tests still pass.

Shape and dtype:
The API returns the same scalar tensor shape `[1]`, dtype `DType::F64`, and `requires_grad = false` for no-grad `mean` and `sum`. The API parity test asserts these properties and bit-exact values.

Final source state:
The candidate source hunks were removed after the final benchmark regressed; `git diff HEAD^ -- crates/ft-api/src/lib.rs crates/ft-kernel-cpu/src/lib.rs` is empty in the rejection commit.

## Gates

- `cargo test -p ft-kernel-cpu gaussian_nll_reduced_f64 -- --nocapture`: passed, 2 tests.
- `cargo test -p ft-api gaussian_nll_loss -- --nocapture`: passed, 4 tests.
- `cargo check -p ft-api -p ft-kernel-cpu --all-targets`: passed; three pre-existing `unused_mut` warnings in recurrent tests remain.
- `cargo fmt -p ft-api -p ft-kernel-cpu --check`: emitted broad pre-existing formatting drift across benches and large existing source sections; no formatter was applied in this lane.
- `cargo clippy -p ft-api -p ft-kernel-cpu --all-targets -- -D warnings`: failed on existing `ft-api` lint debt, ending with 207 errors unrelated to the new reduction helper.
- `ubs crates/ft-api/src/lib.rs crates/ft-kernel-cpu/src/lib.rs`: interrupted after about 5 minutes because UBS hung in `ast-grep` over its shadow workspace; the timeout is recorded in `ubs_ft_api_kernel.txt`.
- `git diff --check`: passed.
- Final `git diff HEAD^ -- crates/ft-api/src/lib.rs crates/ft-kernel-cpu/src/lib.rs`: empty; no sub-threshold code kept.
- `br sync --flush-only`: refused because the local DB is stale and would drop peer issue `frankentorch-ib63`; `.beads/issues.jsonl` already contains `frankentorch-7f1m` closed and was preserved without forcing a lossy export.

## Evidence

See:

- `baseline_4aafb31d.txt`
- `after_4aafb31d_candidate.txt`
- `baseline_rebased_parent_0d2bb62b.txt`
- `after_rebased_head_21f4cfbc.txt`
- `baseline_final_parent_452f41c0.txt`
- `after_final_rebased_head.txt`
- `test_ft_kernel_cpu_gaussian_nll_reduced.txt`
- `test_ft_api_gaussian_nll_loss.txt`
- `check_ft_api_kernel.txt`
- `fmt_check.txt`
- `clippy_ft_api_kernel.txt`
- `ubs_ft_api_kernel.txt`
- `diff_check.txt`
- `final_source_diff.txt`
- `br_sync_flush.txt`
- `evidence.sha256`
