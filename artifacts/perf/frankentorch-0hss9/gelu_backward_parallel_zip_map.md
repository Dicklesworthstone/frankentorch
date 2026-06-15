# frankentorch-0hss9: Gelu backward parallel zip-map

## Target

Crate: `ft-autograd`

Profile-backed family: non-Mish activation backward contribution maps from
`activation_backward/*_chain_16x65536`.

Kept lever: route only `TensorNodeOp::Gelu` first-order backward contribution
generation through the existing order-preserving `tensor_backward_zip_map`
helper. Other non-Mish activation rows stay serial because the broad-family and
zero-allocation attempts were flat or regressed.

## Baseline

Clean worktree: `9124719c`

Worker: `ovh-a`

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a rch exec -- cargo bench -j 1 -p ft-autograd --bench backward_bench -- activation_backward/gelu_chain_16x65536 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Artifact: `artifacts/perf/frankentorch-0hss9/baseline_gelu_ovh_a_clean_head.log`

Row:

- `activation_backward/gelu_chain_16x65536`: `[26.814 ms 27.483 ms 28.719 ms]`

## Candidate

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a rch exec -- cargo bench -j 1 -p ft-autograd --bench backward_bench -- activation_backward/gelu_chain_16x65536 --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Artifact: `artifacts/perf/frankentorch-0hss9/candidate_gelu_parallel_zip_ovh_a.log`

Row:

- `activation_backward/gelu_chain_16x65536`: `[18.156 ms 18.633 ms 19.667 ms]`

Median speedup: `27.483 / 18.633 = 1.48x` (`32.2%` faster).

Rejected probes retained as routing evidence:

- `candidate_non_mish_zero_alloc_activation_backward.log`: zero-allocation
  accumulate over the six non-Mish rows regressed most rows.
- `candidate_softplus_zero_alloc_ovh_a.log`: narrowed Softplus zero-allocation
  was flat (`17.235 ms -> 17.152 ms`, `1.005x`).

## Behavior Proof

- Ordering: `rayon` indexed collect preserves output element order; the existing
  serial `accumulate_tensor_gradient` still accumulates into the target gradient
  in ascending index order.
- Floating point: every Gelu gradient element evaluates the same expression with
  the same incoming gradient and input value. There is no reduction, tie-break,
  RNG, or cross-element dependency.
- Bit proof: `tensor_gelu_large_backward_matches_serial_formula_bit_exact`
  crosses the helper's parallel threshold and compares every gradient bit
  against the prior serial Gelu derivative formula.
- Golden SHA: `sha256sum -c artifacts/optimization/golden_checksums.txt
  --ignore-missing` passed.

## Validation

- `cargo test -j 1 -p ft-autograd tensor_gelu_large_backward_matches_serial_formula_bit_exact -- --nocapture`: passed.
- `cargo test -j 1 -p ft-autograd`: passed, `466 passed`.
- `cargo check -j 1 -p ft-autograd --all-targets`: passed.
- `cargo clippy -j 1 -p ft-autograd --all-targets -- -D warnings`: passed.
- `cargo fmt -p ft-autograd --check`: passed via `rch exec` local/non-compilation path.
- `ubs crates/ft-autograd/src/lib.rs`: returned the pre-existing broad
  ft-autograd warning inventory; clippy/check/test inside UBS were clean.

## Decision

Kept. Score: `Impact 3.0 x Confidence 4.5 / Effort 2.0 = 6.75`.
