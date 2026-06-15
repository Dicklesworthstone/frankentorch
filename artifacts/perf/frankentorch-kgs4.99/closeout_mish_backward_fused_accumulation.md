# frankentorch-kgs4.99 closeout

Target: `activation_backward/mish_chain_16x65536`

Baseline:
- Worker: `vmi1149989`
- Command: `cargo bench -j 1 -p ft-autograd --bench backward_bench -- activation_backward/mish_chain_16x65536 --warm-up-time 1 --measurement-time 5 --sample-size 20 --noplot`
- Time: `[27.166 ms 31.971 ms 39.616 ms]`
- Log: `artifacts/perf/frankentorch-autograd-reprofile-20260615/baseline_mish_current.log`
- Log sha256: `46758c4df293f73bc221dbf4fe03b93505c39c8abba776717b8c735e4cc23956`

Candidate:
- Worker: `vmi1149989`
- Time: `[26.346 ms 27.755 ms 29.380 ms]`
- Log: `artifacts/perf/frankentorch-kgs4.99/candidate_mish_vmi1149989.log`
- Median speedup: `31.971 / 27.755 = 1.1519x`
- Median improvement: `13.19%`
- Score: `3.0 impact * 0.95 confidence / 1.0 effort = 2.85`

Isomorphism:
- The Mish derivative expression is unchanged.
- Each element still applies `target[i] += mish_contribution[i]`; the branch proof pins that FP grouping as `existing_grad + (tsp + x * sig * (1 - tsp * tsp))`.
- Queue/dependency order, hook execution order, create-graph path, tie behavior, and RNG behavior are unchanged.
- The temporary contribution Vec/readback pass is removed from the regular Mish backward accumulation path.

Proof:
- `cargo test -j 1 -p ft-autograd tensor_mish_ -- --nocapture`
- `cargo test -j 1 -p ft-autograd tensor_register_hook -- --nocapture`
- `cargo test -j 1 -p ft-autograd create_graph_first_order_matches_regular_backward -- --nocapture`
- `cargo fmt -p ft-autograd --check`
- `cargo check -j 1 -p ft-autograd --all-targets`
- `cargo clippy -j 1 -p ft-autograd --all-targets -- -D warnings`
- `sha256sum -c artifacts/optimization/golden_checksums.txt`
- `ubs crates/ft-autograd/src/lib.rs`

Verdict: keep.
