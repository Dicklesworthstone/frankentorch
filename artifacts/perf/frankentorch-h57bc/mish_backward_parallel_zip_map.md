# frankentorch-h57bc: Mish backward parallel zip-map

## Target

- Bead: `frankentorch-h57bc`
- Crate: `ft-autograd`
- Profile-backed family: activation backward contribution maps
- Kept lever: order-preserving Rayon zip-map for the compute-heavy `TensorNodeOp::Mish` first-order backward contribution buffer only.

The first broad attempt applied the helper to `gelu/silu/elu/erf/erfc/softplus/mish`. Same-worker evidence showed `mish` was a clear win, but several lighter maps regressed from Rayon overhead. The final source keeps only the `mish` hotspot on the helper.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never rch exec -- cargo bench -j 1 -p ft-autograd --bench backward_bench -- activation_backward --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Artifact: `artifacts/perf/frankentorch-h57bc/baseline_activation_backward.log`

- Worker: `ovh-a`
- `mish_chain_16x65536`: `[64.088 ms 71.142 ms 78.950 ms]`
- Other family medians captured in the baseline log for routing.

## Candidate

Command:

```bash
RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never rch exec -- cargo bench -j 1 -p ft-autograd --bench backward_bench -- activation_backward --warm-up-time 1 --measurement-time 3 --sample-size 10 --noplot
```

Artifact: `artifacts/perf/frankentorch-h57bc/candidate_activation_backward_r3_mish_only.log`

- Worker: `ovh-a`
- `mish_chain_16x65536`: `[27.471 ms 28.765 ms 30.276 ms]`
- Median speedup: `71.142 / 28.765 = 2.47x`

Rejected broad-family artifact:

- `artifacts/perf/frankentorch-h57bc/candidate_activation_backward_r2.log`
- Result: `mish` improved, but lighter rows regressed; not retained.

## Behavior Proof

- Ordering: `rayon` collect preserves output index order for the independent zip-map; gradient accumulation remains serial in ascending target order.
- Floating point: each element evaluates the same Mish derivative expression with the same upstream grad and input value; there is no reduction, tie-breaking, RNG, or cross-element dependency.
- Bit proof: `tensor_mish_large_backward_matches_serial_formula_bit_exact` crosses the parallel threshold and compares every gradient bit against the prior serial formula.
- Golden SHA: `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed.

## Validation

- `cargo test -j 1 -p ft-autograd tensor_mish_large_backward_matches_serial_formula_bit_exact -- --nocapture`: passed.
- `cargo test -j 1 -p ft-autograd`: passed, `465 passed`.
- `cargo check -j 1 -p ft-autograd --all-targets`: passed.
- `cargo clippy -j 1 -p ft-autograd --all-targets -- -D warnings`: passed.
- `cargo fmt -p ft-autograd --check`: passed locally. RCH refused remote fmt because it classifies `cargo fmt` as non-compilation when remote is required.
- `ubs crates/ft-autograd/src/lib.rs crates/ft-autograd/benches/backward_bench.rs`: returned pre-existing broad crate findings; rustfmt/clippy/check/test inside UBS were clean.

## Decision

Kept. The final lever is isolated to the measured Mish backward hotspot and clears the `Score >= 2.0` threshold with a same-worker `2.47x` median win.
