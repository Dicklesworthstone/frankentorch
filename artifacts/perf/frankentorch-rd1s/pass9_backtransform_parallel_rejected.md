# frankentorch-rd1s pass 9: back-transform row-parallel update rejected

## Target

- Bead: `frankentorch-rd1s`
- Surface: `ft-kernel-cpu` full-vector `eigh`
- Profile-backed residual: current same-worker `ts1` split shows full `eigh_f64_256x256`
  still pays vector/back-transform work above values-only `eigvalsh_f64_256x256`.

## Baseline

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts2 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- 'eigh_f64_256x256|eigvalsh_f64_256x256' --warm-up-time 1 --measurement-time 5 --sample-size 20
```

RCH selected worker `ts1`.

Results:

- `eigh_f64_256x256`: `[10.108 ms 10.222 ms 10.331 ms]`
- `eigvalsh_f64_256x256`: `[6.8096 ms 6.8576 ms 6.8968 ms]`

## Candidate

One lever tested: thresholded Rayon row-parallel update in
`eigh_tred2_backtransform` after the serial projection accumulation. The update
is row-independent, so the intended proof was exact per-element arithmetic with
only row scheduling changed.

## Proof Gate

Focused tests passed:

- `cargo test -p ft-kernel-cpu eigvalsh_matches_eigh -- --nocapture`
- `cargo test -p ft-kernel-cpu eigh_tred2_tql2_orthonormal_and_reconstructs_24x24 -- --nocapture`

Clean golden payloads were bit-identical:

- before: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- after: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`

Isomorphism notes:

- Ordering preserved: yes, final `f64::total_cmp` sort unchanged.
- Tie-breaking unchanged: yes, candidate did not touch the eigen-pair sort.
- Floating-point: intended bit-identical per element; clean golden confirmed.
- RNG: N/A.

## Benchmark Gate

Command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-kernel-cpu --bench linalg_bench -- 'eigh_f64_256x256|eigvalsh_f64_256x256' --warm-up-time 1 --measurement-time 5 --sample-size 20
```

RCH selected worker `ts1`.

Candidate results:

- `eigh_f64_256x256`: `[68.866 ms 82.040 ms 97.793 ms]`
- `eigvalsh_f64_256x256`: `[10.152 ms 10.322 ms 10.514 ms]`

RCH artifact sync also surfaced a second candidate run on worker `ts2`:

- `eigh_f64_256x256`: `[85.747 ms 88.124 ms 90.484 ms]`
- `eigvalsh_f64_256x256`: `[10.307 ms 10.350 ms 10.413 ms]`

Verdict: rejected. Rayon fan-out per Householder step catastrophically regressed
the full-vector path and also polluted the values-only comparison. Source hunk
removed.

## Next Primitive

Do not retry row fan-out around the EISPACK stream. The next rd1s lever is the
deeper no-gaps primitive: blocked `dsytrd` panel plus compact-WY/trailing
rank-2k update, with strict fallback to the current scalar path unless golden
and ordering/sign ledgers pass.
