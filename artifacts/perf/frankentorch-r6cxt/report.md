# frankentorch-r6cxt - flat recurrent sequence workspace

## Target

- Bead: `frankentorch-r6cxt`
- Title: `[perf][no-gaps] ft-api flat recurrent sequence workspace`
- Crate surface: `ft-api`
- Benchmark group: `recurrent_forward`
- Worker discipline: same-worker `rch`, crate-scoped only

## Prior evidence

Follow-up from `frankentorch-09si4`:

- persistent packed RHS replay was rejected because `matrixmultiply` packed APIs
  are private and scalar replay already failed bit-exact proof;
- borrowed immutable recurrent tensor storage passed focused checks/goldens while
  present but regressed same-worker `fmd` medians;
- next non-panel primitive: flatten recurrent sequence/output workspaces to remove
  `Vec<Vec<f64>>` staging, `outputs.clone()` between layers, and final flatten
  copies while preserving unchanged `dgemm_bt` arithmetic.

## Pass 1 - baseline

Baseline command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=fmd RCH_WORKERS=fmd rch exec -- \
  cargo bench -p ft-api --bench ops_bench -- recurrent_forward \
  --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `fmd`.

Baseline medians:

| Case | Baseline |
| --- | ---: |
| `lstm_seq64_batch1_128x128` | `3.2712 ms` |
| `gru_seq64_batch1_128x128` | `2.8212 ms` |
| `rnn_tanh_seq64_batch1_128x128` | `705.89 us` |

## Pass 2-3 - primitive and proof plan

Candidate: replace per-timestep `Vec<Vec<f64>>` recurrent staging with flat
contiguous sequence/output workspaces.

The lever preserves:

- layer/direction/timestep/batch/feature order by using the same row-major
  `t * batch * width + b * width + i` indexing;
- `dgemm_bt` arithmetic, dimensions, and RHS/LHS slices;
- gate equations and bias add grouping;
- RNG absence.

Proof gate: focused recurrent raw-forward golden tests plus golden SHA-256
verification before same-worker rebench.

## Pass 3 - one-lever implementation

Implemented one source lever in `crates/ft-api/src/lib.rs`:

- build initial recurrent input as one flat contiguous sequence workspace instead
  of `Vec<Vec<f64>>`;
- write layer outputs into one flat output workspace with the same
  `t * batch * out_size + direction_offset` order;
- rotate workspaces between stacked layers with `swap`, then resize the output
  workspace for the next layer;
- pass the final output workspace directly to `tensor_variable`.

No matrixmultiply helper, gate equation, layer traversal, direction traversal, or
RNG behavior changed.

## Pass 4 - behavior proof

Focused recurrent raw-forward golden command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=fmd RCH_WORKERS=fmd rch exec -- \
  cargo test -p ft-api raw_forward_golden_isomorphism -- --nocapture
```

Result on `fmd`: passed 3/3
(`lstm_raw_forward_golden_isomorphism`,
`gru_raw_forward_golden_isomorphism`,
`rnn_raw_forward_golden_isomorphism`).

Golden SHA-256 command:

```bash
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
```

Result: all present tracked golden outputs passed.

Compile command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=fmd RCH_WORKERS=fmd rch exec -- \
  cargo check -p ft-api --all-targets
```

Result: passed on `fmd`.

Isomorphism checklist:

- ordering: preserved exact timestep, batch, feature, layer, and direction
  output order;
- tie-breaking: no comparisons or selection behavior changed;
- floating point: unchanged `dgemm_bt` calls, gate equations, and add/mul order
  inside each recurrent step;
- RNG: none introduced or moved;
- output goldens: recurrent raw-forward goldens and tracked SHA-256 ledger passed.

## Pass 5 - same-worker rebench

After command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=fmd RCH_WORKERS=fmd rch exec -- \
  cargo bench -p ft-api --bench ops_bench -- recurrent_forward \
  --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `fmd`.

| Case | Baseline | First after | Speedup |
| --- | ---: | ---: | ---: |
| `lstm_seq64_batch1_128x128` | `3.2712 ms` | `3.1111 ms` | `1.051461x` |
| `gru_seq64_batch1_128x128` | `2.8212 ms` | `2.2303 ms` | `1.264942x` |
| `rnn_tanh_seq64_batch1_128x128` | `705.89 us` | `726.76 us` | `0.971284x` |

Geomean speedup: `1.089105x`.

Provisional score after this first run was above gate, but a current-tree
re-run was required after `HEAD` advanced and the masked-SDPA compile blocker
was repaired.

Current-tree after run:

| Case | Baseline | Current after | Speedup |
| --- | ---: | ---: | ---: |
| `lstm_seq64_batch1_128x128` | `3.2712 ms` | `3.8165 ms` | `0.857120x` |
| `gru_seq64_batch1_128x128` | `2.8212 ms` | `2.2682 ms` | `1.243806x` |
| `rnn_tanh_seq64_batch1_128x128` | `705.89 us` | `763.97 us` | `0.923976x` |

Current geomean: `0.994989x`.

Confirmation run:

| Case | Baseline | Confirm after | Speedup |
| --- | ---: | ---: | ---: |
| `lstm_seq64_batch1_128x128` | `3.2712 ms` | `3.9669 ms` | `0.824624x` |
| `gru_seq64_batch1_128x128` | `2.8212 ms` | `3.0162 ms` | `0.935349x` |
| `rnn_tanh_seq64_batch1_128x128` | `705.89 us` | `1.0383 ms` | `0.679852x` |

Confirmation geomean: `0.806395x`.

Score: `0.0`, below the `>= 2.0` keep gate.

## Pass 6 - closeout and shifted target

Verdict: REJECTED. The recurrent flat-workspace source hunk was removed.

Residual gates:

- `git diff --check` passed for the staged source/evidence files;
- package-level `cargo fmt --check --package ft-api` and file-level rustfmt
  checks show broad pre-existing ft-api formatting drift, so no formatting files
  were changed in this one-lever commit;
- `cargo clippy -p ft-api --all-targets -- -D warnings` remains blocked by
  existing ft-api lint debt, including lints in recurrent functions that predate
  this source hunk;
- `ubs crates/ft-api/src/lib.rs` completed locally with broad existing ft-api
  inventory; no new unsafe code was introduced.

Next shifted primitive: reprofile `recurrent_forward` from the retained baseline
path and attack a materially different hidden-projection primitive. Do not
repeat packed-panel replay, scalar replay, branch splitting, padded-M panels,
borrowed recurrent storage, or flat sequence/output workspaces.
