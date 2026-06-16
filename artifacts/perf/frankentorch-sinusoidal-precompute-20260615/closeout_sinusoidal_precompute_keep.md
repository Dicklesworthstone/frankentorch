# frankentorch-tn0py closeout: sinusoidal PE denominator precompute

## Lever

- Target: `ft-api::FrankenTorchSession::sinusoidal_positional_encoding`.
- Profile-backed symptom: `sinusoidal_pe/4096x1024` spent most time recomputing `10000.powf((2*i)/d_model)` for every row/position.
- Change kept in `main`: precompute per-frequency denominators once, then fill rows in parallel with the same `angle = pos as f64 / denominator[i]` expression and unchanged sin/cos column ordering.
- Source commit containing the hunk: `7a924b10 perf(ft-api,ft-kernel-cpu): direct depthwise conv2d kernel, no-grad fast path (5.7-8.7x)`.

## Benchmark Evidence

All benchmark commands were RCH crate-scoped:

```text
rch exec -- cargo bench -j 1 -p ft-api --bench ops_bench -- 'rope_freqs/32768x128|sinusoidal_pe/4096x1024' --warm-up-time 1 --measurement-time 5 --sample-size 20 --noplot
```

Initial profile-backed baseline on `vmi1152480`:

```text
rope_freqs/32768x128    [43.618 ms 52.030 ms 61.330 ms]
sinusoidal_pe/4096x1024 [41.754 ms 47.759 ms 53.627 ms]
```

Candidate rebench on `vmi1152480`:

```text
rope_freqs/32768x128    [15.398 ms 17.987 ms 21.691 ms]
sinusoidal_pe/4096x1024 [17.980 ms 20.427 ms 23.265 ms]
```

Tighter old/new pair observed on `vmi1149989`:

```text
old sinusoidal_pe/4096x1024       [12.850 ms 13.753 ms 14.567 ms]
candidate sinusoidal_pe/4096x1024 [ 8.264 ms  8.530 ms  8.742 ms]
```

Target-row improvement from the same-worker pair: `13.753 ms -> 8.5298 ms`, `1.61x`.
The control row moved substantially across repeated RCH runs, so confidence is not rated as a clean isolated microbenchmark. The algorithmic win is still direct: per-call powf count drops from `max_len * ceil(d_model / 2)` to `ceil(d_model / 2)`, and the output is bit-identical.

Score: `Impact 1.61 * Confidence 1.25 / Effort 0.75 = 2.68`, keep.

## Isomorphism Proof

Invariant coverage:

- Ordering preserved: row-major `[max_len, d_model]`, even columns `sin`, odd columns `cos`.
- Floating point preserved: each element uses the same denominator value as the old per-element `powf` expression; golden test compares `to_bits()`.
- Odd width preserved: final odd column is the sine lane for `i = d_model / 2`.
- RNG preserved: no RNG use.
- Error behavior preserved: `d_model == 0` still returns the same incompatible-set error before allocation.

Golden digest artifact:

```text
golden_sinusoidal_pe_digest.txt sha256:
37b8ddd6ceb49ca7cc775353c40ba87da059e397fa9733540c3a1427879fde39

digest asserted by test:
0x3cc56e1b0a69108c
```

Focused proof passed:

```text
rch exec -- cargo test -j 1 -p ft-api sinusoidal_position_encoding_parallel_match_serial_bit_exact --lib
vmi1152480: 1 passed; 0 failed; 2119 filtered out.
```

Compile gate passed:

```text
rch exec -- cargo check -j 1 -p ft-api --lib --tests
vmi1149989: finished clean.
```

Lint gate status:

- `rch exec -- cargo clippy -j 1 -p ft-api --lib --tests -- -D warnings` failed before this patch in `ft-autograd` at `crates/ft-autograd/src/lib.rs:9708` (`clippy::manual_checked_ops`).
- `rch exec -- cargo clippy -j 1 -p ft-api --lib --tests --no-deps -- -D warnings` then failed on existing `ft-api` lint debt (258 errors across old public APIs, documentation indentation, pre-existing coefficient table precision, and similar).
- `ubs crates/ft-api/src/lib.rs` returned exit 0 and reported its own internal fmt/clippy/check/test sections clean, but also inventoried broad historical panic/security/perf warnings across the full large file. No UBS finding is localized to the sinusoidal hunk.

Follow-up policy: do not broaden this perf lever into the unrelated lint cleanup; track that separately.
