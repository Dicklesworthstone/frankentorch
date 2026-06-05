# frankentorch-0ijz - fused no-grad pdist p!=2

Target: `ft-api` / `ft-kernel-cpu` no-grad `tensor_pdist` for finite `p > 0, p != 2` and `p = +inf`.

Profile-backed hotspot:

- `frankentorch-hnhq` proved the same op-graph shape for `cdist`: the p!=2 no-grad path materialised a full broadcast difference tensor before reduction.
- `tensor_pdist` had the same residual for row pairs: `index_select + sub + abs + pow + sum_dim + pow` materialised `[out_len, M]` where `out_len = N*(N-1)/2`.
- The bench harness compares the old materialised path (`pdist_p1_broadcast`) with the production no-grad fused path (`pdist_p1_fused`).

One lever:

- Added `ft_kernel_cpu::pdist_forward_f64`, streaming each strict-upper-triangle row pair in one pass and writing the flattened `(i, then j)` order.
- Routed only no-grad p!=2 finite positive and `+inf` cases through the fused kernel.
- p=2 still uses the existing matmul identity; all grad cases still use the autograd op graph.

Same-worker RCH Criterion (`ts1`):

- Command: `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-api --bench cdist_bench -- pdist_p1 --warm-up-time 1 --measurement-time 5 --sample-size 20`
- `pdist_p1_broadcast/256x128`: `[26.789 ms 27.139 ms 27.520 ms]`
- `pdist_p1_fused/256x128`: `[2.9553 ms 3.0369 ms 3.1623 ms]`
- `pdist_p1_broadcast/512x64`: `[53.487 ms 56.562 ms 60.450 ms]`
- `pdist_p1_fused/512x64`: `[9.3464 ms 10.477 ms 11.487 ms]`
- Median speedups: `8.94x` and `5.40x`.
- Score: `Impact 4.0 x Confidence 4.0 / Effort 1.5 = 10.7`, keep.

Behavior proof:

- Ordering: output remains PyTorch-style flattened strict upper triangle, nested `for i` then `for j in i+1..n`.
- Tie-breaking: no tie selection exists in this distance reduction; `p=+inf` preserves first-observed max value semantics because only the final max value is observable.
- Floating point: finite p performs the same per-pair per-k subtraction, abs, `powf(p)`, serial accumulation, and final `powf(1/p)` order as the materialised op graph. `p=+inf` performs the same per-k max-abs scan order.
- RNG: no random state or draw order is touched.
- Autograd: grad path unchanged because fused routing is gated by `!tensor_requires_grad(input)`.
- Golden SHA-256: `eb44a7cf564f77fea6098abeec5f853bde239e6c7b5c18307f21178ee55d7a85` for `artifacts/perf/frankentorch-0ijz/pdist_golden_reference_and_fused.txt`.

Verification:

- `RCH_REQUIRE_REMOTE=1 rch exec -- env FT_PDIST_GOLDEN_OUT=artifacts/perf/frankentorch-0ijz/pdist_golden_reference_and_fused.txt cargo run -p ft-api --example pdist_golden`: passed, reference-vs-fused assertions executed.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`: passed, including the pdist golden artifact.
- `RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo test -p ft-api pdist_p_neq2_fused_nograd_matches_broadcast_bit_exact -- --nocapture`: passed.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-api -p ft-kernel-cpu --all-targets`: passed.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings`: passed.
- `rch exec -- cargo fmt -p ft-api -p ft-kernel-cpu --check`: blocked by pre-existing package-wide formatting drift in unrelated `ft-api` benches; the touched pdist bench/example formatting was normalized manually.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-api -p ft-kernel-cpu --all-targets -- -D warnings`: blocked by pre-existing broad `ft-api/src/lib.rs` warning inventory outside this lever; `ft-kernel-cpu` clippy passed separately.

Residual:

- The next deeper primitive is still `frankentorch-84li`: blocked symmetric tridiagonalization plus tridiagonal divide-and-conquer/secular merge for the full `eigh` residual.
