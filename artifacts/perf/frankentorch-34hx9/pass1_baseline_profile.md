# frankentorch-34hx9 pass 1: baseline and profile

## Scope

- Bead: `frankentorch-34hx9`
- Target: blocked `dsytrd` tridiagonalization for `eigh`
- Source edits: none
- Benchmark crate: `ft-kernel-cpu`
- RCH policy: remote required, crate-scoped, `cargo -j 1`

## Bench Rows Inspected

Current `crates/ft-kernel-cpu/benches/linalg_bench.rs` contains:

- `eigh_f64_256x256`
- `eigvalsh_f64_256x256`
- `sym_rank2k_lower_scalar_f64_256x32`
- `sym_rank2k_lower_gemm_f64_256x32`

The existing `rd1s` rank-2k evidence already established the primitive shape:
`A := A - (V @ W^T + W @ V^T)` over the lower triangle only. The latest valid
prior artifact row was on worker `vmi1227854`, with scalar `[792.39 us 823.98 us
864.56 us]` and GEMM `[257.67 us 260.74 us 264.13 us]` for a `3.16x` median
primitive win. This pass re-ran the current tree instead of relying on that
cross-worker row.

## RCH Baseline: `eigh` / `eigvalsh`

Command:

```bash
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- 'eigh_f64_256x256|eigvalsh_f64_256x256' --warm-up-time 1 --measurement-time 5 --sample-size 20 2>&1 | tee artifacts/perf/frankentorch-34hx9/pass1_baseline_eigh.log
```

RCH selected worker `ovh-a` at `ubuntu@51.222.245.56`.

Rows:

- `eigh_f64_256x256`: `[8.9996 ms 9.0613 ms 9.1502 ms]`
- `eigvalsh_f64_256x256`: `[5.6935 ms 5.7728 ms 5.8773 ms]`

The full-vector path is `1.57x` the values-only path by median on this worker,
with a `3.2885 ms` median gap.

## RCH Baseline: Rank-2k Primitive

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- 'sym_rank2k_lower_(scalar|gemm)_f64_256x32' --warm-up-time 1 --measurement-time 5 --sample-size 20 2>&1 | tee artifacts/perf/frankentorch-34hx9/pass1_baseline_rank2k.log
```

RCH selected worker `ovh-a` at `ubuntu@51.222.245.56`.

Rows:

- `sym_rank2k_lower_scalar_f64_256x32`: `[1.2448 ms 1.2644 ms 1.2860 ms]`
- `sym_rank2k_lower_gemm_f64_256x32`: `[266.05 us 280.49 us 300.34 us]`

The GEMM-backed lower rank-2k primitive is `4.51x` faster by median on the same
worker. This confirms the blocked trailing-update building block is productive.

## RCH Profile-Time Attempt

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ovh-a rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- eigh_f64_256x256 --profile-time 5 2>&1 | tee artifacts/perf/frankentorch-34hx9/pass1_profile_time_eigh.log
```

RCH selected worker `ovh-a` at `ubuntu@51.222.245.56`.

Result: Criterion completed `eigh_f64_256x256` profiling for `5.0000 s`
successfully, with analysis disabled as expected for `--profile-time`.

## Productive Target for Pass 2

Pass 1 is productive. RCH did not refuse remote execution, and every measured
row came from the same worker.

Use `eigh_f64_256x256|eigvalsh_f64_256x256` as the public-path benchmark pair
and `sym_rank2k_lower_(scalar|gemm)_f64_256x32` as the primitive guardrail.
Pass 2 should attack the blocked `dsytrd` panel integration around the current
serial Householder tridiagonalization, using the already-measured GEMM lower
rank-2k update as the trailing-update lever. Do not retry row fan-out around the
EISPACK stream; prior `rd1s` evidence showed that family regresses.

Proof obligations for the first implementation pass:

- Strict-mode reference fallback remains available unless the blocked route
  passes the accepted floating-point ledger.
- Ordering and tie-breaking are preserved by the existing final eigenpair sort.
- Eigenvector sign/orientation and reconstruction tolerance are checked against
  the existing golden/oracle path.
- Golden SHA is recorded before and after any source change.
