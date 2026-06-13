# frankentorch-kgs4.46 pass 1: packed B-panel f64 GEMM keep

Bead: `frankentorch-kgs4.46`

Target: profile-backed `ft-kernel-cpu` f64 square GEMM wall after the kgs4.45 2-D tiling work.

Primitive: Goto/BLIS-style communication reduction. For the f64 normal 2-D GEMM path, each output column panel now copies the logical `B[:, j0..j1]` panel once into contiguous row-major scratch and reuses it across all M tiles in that N panel.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 CARGO_TERM_COLOR=never rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench gemm_bench matmul_f64_512x512x512 -- --warm-up-time 1 --measurement-time 3 --sample-size 10
```

RCH selected worker: `vmi1227854`

Criterion:

```text
matmul_f64_512x512x512  time:   [3.9236 ms 4.2830 ms 4.7175 ms]
```

Artifact: `pass1_baseline_criterion_f64_512_vmi1227854.log`

## One lever

Changed only `dgemm_2d_parallel` in `crates/ft-kernel-cpu/src/lib.rs`.

Before: each `(M tile, N tile)` task passed a strided view of the original B matrix into `matrixmultiply`.

After: each N tile packs `B[:, j0..j1]` into a contiguous `[k, bj]` panel once, then all M tiles in that N panel call the same GEMM microkernel over the packed panel.

## Isomorphism proof

- Output partitioning: unchanged disjoint 2-D C tiles.
- K accumulation: K is never split; every output element is still computed by one GEMM call over the full K range.
- Floating point: copied B values preserve the exact logical operand stream for each `(i, j, k)`; the focused bit-exact test and golden SHA prove no observable FP drift.
- Ordering and tie-breaking: GEMM has no comparison or tie surface.
- RNG: no RNG use in the path.
- Safety: B panel scratch is immutable during nested M-tile work; C tile writes remain disjoint via the existing `TilePtr` invariant.

Proof commands:

```text
RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never rch exec -- cargo test -j 1 -p ft-kernel-cpu gemm_2d_parallel_is_bit_exact_vs_serial -- --nocapture
RCH_REQUIRE_REMOTE=1 CARGO_TERM_COLOR=never rch exec -- cargo run --release -q -p ft-kernel-cpu --example gemm_golden
```

Proof results:

```text
gemm::tile_iso_tests::gemm_2d_parallel_is_bit_exact_vs_serial ... ok
test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 451 filtered out
```

Golden stdout SHA-256:

```text
baseline  83ef409d8780f17596dfeef1d2b3c406c06d5c354b0ce691d847af32936af160
candidate 83ef409d8780f17596dfeef1d2b3c406c06d5c354b0ce691d847af32936af160
```

## Rebench

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 CARGO_TERM_COLOR=never rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench gemm_bench matmul_f64_512x512x512 -- --warm-up-time 1 --measurement-time 3 --sample-size 10
```

RCH selected worker: `vmi1227854`

Criterion:

```text
matmul_f64_512x512x512  time:   [3.1800 ms 3.4499 ms 3.8373 ms]
```

Median speedup: `4.2830 / 3.4499 = 1.241x`.

Score: `2.48 = Impact 1.241 * Confidence 0.90 / Effort 0.45`.

Verdict: KEEP. The same-worker speedup clears Score >= 2.0.

## Gates

- `cargo check -j 1 -p ft-kernel-cpu --lib --examples --benches`: passed on RCH `vmi1149989`; existing `gemm_golden.rs` warnings only.
- `cargo clippy -j 1 -p ft-kernel-cpu --lib -- -D warnings`: passed on RCH `vmi1149989`.
- `cargo test -j 1 -p ft-kernel-cpu gemm_2d_parallel_is_bit_exact_vs_serial -- --nocapture`: passed on RCH `vmi1149989`.
- `git diff --check`: passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs`: 0 critical findings.
- `cargo fmt -p ft-kernel-cpu --check`: blocked by pre-existing unformatted example files in this worktree; no formatting changes were made outside the source lever.

## Next profile route

Re-profile after this keep. Residual GEMM lanes should measure either larger f64 shapes (`1024+`) or f32/BT surfaces separately; do not repeat plain packed-panel variants without a new profile-backed target.
