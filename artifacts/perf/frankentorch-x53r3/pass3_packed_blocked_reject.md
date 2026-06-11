# Pass 3 Rejection: Packed-Lower Blocked WY Harness

## Lever

Prototype a packed-lower-storage variant of the experimental blocked dsytrd
harness behind `eigvalsh_blocked_f64`:

- keep production `eigvalsh_contiguous_f64` unchanged;
- keep the same blocked WY panel math as `eigh_tridiag_reduce_blocked`;
- store/copy/update only lower packed storage in the blocked harness.

The source hunk was removed after the rebench.

## Baseline

Baseline command:

```bash
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -v -- cargo run --release -j 1 -p ft-kernel-cpu --example eigvalsh_blocked_ab
```

The first attempt was refused by RCH while another frankentorch job owned the
project slot:

- artifact: `pass3_blocked_ab_baseline_vmi1227854.log`
- SHA256: `403229703826c47cc989e6c7b9d93c45c0c43099fc50ac8301c53d427d3a4a6f`
- refusal: `critical_pressure=1,active_project_exclusion=1`

Retry baseline on `vmi1227854`:

| n | packed scalar | full blocked | ratio |
| ---: | ---: | ---: | ---: |
| 256 | `6.319 ms` | `6.086 ms` | `1.04x` |
| 512 | `51.316 ms` | `54.024 ms` | `0.95x` |
| 768 | `164.288 ms` | `177.419 ms` | `0.93x` |
| 1024 | `412.726 ms` | `441.702 ms` | `0.93x` |

Baseline artifact:

- `artifacts/perf/frankentorch-x53r3/pass3_blocked_ab_baseline_vmi1227854_retry.log`
- SHA256: `0520cf3b144b1f574df7023cb194defdea8ee91d20ac2a296613f001cae1a37f`

## Candidate

The first candidate run was refused by RCH while another frankentorch job owned
the project slot:

- artifact: `pass3_packed_blocked_ab_candidate_vmi1227854.log`
- SHA256: `11a4c9df4da9bea365234e3a296415238dc769126d48835cf8e792f44a60accd`
- refusal: `critical_pressure=1,active_project_exclusion=1`

Retry candidate on `vmi1227854`:

| n | packed scalar | packed blocked | ratio |
| ---: | ---: | ---: | ---: |
| 256 | `7.164 ms` | `9.159 ms` | `0.78x` |
| 512 | `49.160 ms` | `57.556 ms` | `0.85x` |
| 768 | `163.714 ms` | `199.510 ms` | `0.82x` |
| 1024 | `372.160 ms` | `425.093 ms` | `0.88x` |

Candidate artifact:

- `artifacts/perf/frankentorch-x53r3/pass3_packed_blocked_ab_candidate_vmi1227854_retry.log`
- SHA256: `3eb86dd626efe4cca77f09fa796f02b27bd7f7144ca35dd0ca8e1ec58207f19e`

The A/B harness also reported max sorted-eigenvalue deltas at `~1e-12`, so the
candidate preserved tolerance parity. Ordering stayed sorted by `total_cmp`, and
there was no RNG.

## Decision

Reject. Packed storage around the same blocked WY panel math is slower than the
existing full blocked harness and much slower than the production packed scalar
path. Score is `0`.

Next primitive: replace the panel algorithm itself. The next pass should target
communication-avoiding/bulge-chasing SBR or tridiagonal divide-and-conquer after
the reduction, not another storage wrapper around the same dsytrd panel.
