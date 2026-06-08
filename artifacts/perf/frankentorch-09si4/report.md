# frankentorch-09si4 - recurrent RHS optimization

## Target

- Bead: `frankentorch-09si4`
- Title: `[perf][no-gaps] ft-api persistent prepacked recurrent RHS panels`
- Crate surface: `ft-api`, possibly `ft-kernel-cpu`
- Benchmark group: `recurrent_forward`
- Worker discipline: same-worker `rch`, crate-scoped only
- Starting source: `7839ca1b3a40f522a99db2d8b2585ec7389f5535`

## Pass 1 - baseline and hotspot validation

Baseline command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=yto rch exec -- \
  cargo bench -p ft-api --bench ops_bench -- recurrent_forward \
  --warm-up-time 1 --measurement-time 5 --sample-size 20
```

The command banner named `yto`, but `rch` selected `fmd` in the authoritative
log:

```text
Selected worker: fmd at ubuntu@51.222.245.56
```

Baseline medians on `fmd`:

| Case | Baseline |
| --- | ---: |
| `lstm_seq64_batch1_128x128` | `3.1465 ms` |
| `gru_seq64_batch1_128x128` | `2.2762 ms` |
| `rnn_tanh_seq64_batch1_128x128` | `702.13 us` |

Direct `perf stat` / flamegraph profiling was attempted by the pass-1 subagent,
but `rch` rejected those non-compilation commands under remote-required policy.
The target remains profile-backed by the prior recurrent artifacts and current
source shape:

- input projection is already batched over all timesteps;
- recurrent projection still calls `matmul_rhs_transposed_contiguous_f64_into`
  once per timestep for LSTM, GRU, and RNN;
- nearby families already rejected: materialized recurrent transpose,
  scalar row-vector dot rewrite, gate-branch splitting, and padded single-row
  matrixmultiply panels.

Candidate rebench must use the same `fmd` worker.

## Pass 2-3 - primitive selection and proof gate

Candidate ranking after applying the graveyard numeric-kernel/data-movement
families:

| Rank | Candidate | Score | Verdict |
| ---: | --- | ---: | --- |
| 1 | Persistent packed RHS panel replay for batch-1 recurrent `h @ W_hh^T` | `4 x 3 / 4 = 3.0` | selected for proof gate |
| 2 | Full safe-Rust small-GEMM replacement for all `m == 1` RHS-transposed f64 | `4 x 2 / 5 = 1.6` | too broad and repeats failed FP-order risk |
| 3 | Algebraic recurrent scan / low-rank / structured recurrent transform | `5 x 1 / 5 = 1.0` | not exact for arbitrary weights |

Canonical primitive family:

- `alien_cs_graveyard.md` A1 numeric kernels: cache locality, SIMD, precision.
- `alien_cs_graveyard.md` §9.6 communication-avoiding algorithms: reduce
  data movement in linear algebra kernels.
- `/data/projects/.scratch/no_gaps_directive.txt`: safe-Rust packed-panel GEMM
  and register microkernels, with behavior parity absolute.

Proof-gate result: rejected before source implementation. `matrixmultiply`
does pack B inside every GEMM call, but the needed `pack_nr` and `gemm_packed`
surfaces are crate-private. The only public f64 API is `matrixmultiply::dgemm`,
which accepts raw strided matrices and performs packing internally each call.
Reusing packed RHS panels would therefore require copying private unsafe
internals or changing the arithmetic traversal. The scalar row-vector family
already failed bit-exact proof on `frankentorch-8x2i`, so a safe scalar replay
is not a valid fallback for this bead.

Fallback trigger fired:

- no stable packed-RHS replay helper can be built as one small safe-Rust lever;
- do not try more panel-packing variants in this recurrent lane;
- route next to a non-panel primitive that preserves `dgemm_bt` arithmetic, such
  as borrowed recurrent tensor storage or flat sequence/output layout to reduce
  data movement around the unchanged GEMM calls.

## Packed-RHS implementation feasibility addendum

A verifier pass confirmed the rejection:

- `matrixmultiply` publicly reexports only `dgemm` / `sgemm`;
- `GemmKernel::pack_nr`, `packing::pack` / `pack_avx2`, and `gemm_packed`
  are private or `pub(crate)`;
- exact packed-B replay would need the same runtime-selected microkernel,
  pack layout, masked edge path, and MXCSR restoration wrapper;
- copying/vendor-patching that internal unsafe layer is outside this bead's
  one-lever safe-Rust budget.

Verdict: close the packed-RHS sub-primitive as rejected and move to a non-panel
data-movement primitive around unchanged public `dgemm_bt` calls.

## Pass 4-6 - borrowed recurrent tensor storage rejection

The next exactness-preserving candidate was to borrow immutable contiguous f64
tensor storage from the tape during raw LSTM/GRU/RNN forward instead of cloning
the input, weights, recurrent weights, and biases before the recurrent loop.
This preserves all gate equations, timestep/layer/direction ordering,
floating-point operation order inside GEMM, and RNG absence; it only changes the
ownership path for immutable source tensors.

Verification while the candidate source hunk was present:

- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-autograd -p ft-api --all-targets`
  passed on worker `vmi1149989`.
- `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing`
  passed for locally present golden outputs.
- `RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-api raw_forward_golden_isomorphism -- --nocapture`
  passed 3/3 recurrent raw-forward goldens on worker `fmd`.

Same-worker rebench command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKERS=fmd rch exec -- \
  cargo bench -p ft-api --bench ops_bench -- recurrent_forward \
  --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Candidate medians on `fmd`:

| Case | Baseline | Borrowed storage | Speedup |
| --- | ---: | ---: | ---: |
| `lstm_seq64_batch1_128x128` | `3.1465 ms` | `3.2100 ms` | `0.9802x` |
| `gru_seq64_batch1_128x128` | `2.2762 ms` | `2.3057 ms` | `0.9872x` |
| `rnn_tanh_seq64_batch1_128x128` | `702.13 us` | `742.79 us` | `0.9453x` |

Geomean speedup: `0.9707x`. Score: `0.0`, below the `>= 2.0`
keep gate because the candidate regressed all three selected medians. The source
hunk was removed; no borrowed recurrent storage code is retained.

Final verdict for `frankentorch-09si4`: rejected. Two panel-adjacent families
are now closed off for this lane: persistent packed RHS replay is blocked by
private `matrixmultiply` packing surfaces and scalar replay exactness risk, while
borrowed immutable tensor storage does not improve the measured recurrent
forward benchmark. The next deeper primitive should be a flat recurrent
sequence/output workspace that removes `Vec<Vec<f64>>` sequence materialization,
`outputs.clone()` between layers, and final flatten copies while preserving the
unchanged `dgemm_bt` arithmetic and exact output order.
