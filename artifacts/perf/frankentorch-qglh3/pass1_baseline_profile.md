# frankentorch-qglh3 Pass 1 Baseline/Profile

Bead: `frankentorch-qglh3`
Agent: `IvoryDeer`
Started: 2026-06-11T01:14:00Z

## Claim + Coordination

- `br update frankentorch-qglh3 --status in_progress --assignee IvoryDeer`
- File reservations granted without conflicts for:
  - `crates/ft-kernel-cpu/src/lib.rs`
  - `crates/ft-kernel-cpu/benches/linalg_bench.rs`
  - `crates/ft-kernel-cpu/examples/eigvals_golden.rs`
  - `artifacts/perf/frankentorch-qglh3/**`
  - `.skill-loop-progress.md`
  - `.beads/issues.jsonl`
- Mail thread: `frankentorch-qglh3`

## Remote Criterion Baselines

Worker: `vmi1227854`

Commands:

```bash
env RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- \
  eigvals_f64_256x256 --sample-size 10 --warm-up-time 1 --measurement-time 3

env RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- \
  cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- \
  eig_f64_256x256 --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Results:

| Row | p50/median | Interval |
| --- | ---: | --- |
| `eigvals_f64_256x256` | `27.445 ms` | `[26.514 ms 27.445 ms 29.029 ms]` |
| `eig_f64_256x256` | `49.892 ms` | `[49.080 ms 49.892 ms 51.006 ms]` |

Artifacts:

- `pass1_baseline_eigvals_256.log`
- `pass1_baseline_eig_256.log`

## Golden Anchor

Remote golden attempts were refused by RCH:

- `pass1_eigvals_golden_before.log`: `critical_pressure=1,insufficient_slots=1`
- `pass1_eigvals_golden_before_retry.log`: `critical_pressure=1,active_project_exclusion=1`

Deterministic proof anchor captured through `rch exec` local fallback:


```text
n=64  eigvals_digest=0xbc0583d464b1a211 eig_digest=0xbc0583d464b1a211
n=128 eigvals_digest=0x763c4b15d92c4b89 eig_digest=0x763c4b15d92c4b89
n=256 eigvals_digest=0x00b87b4996340204 eig_digest=0x00b87b4996340204
```

SHA-256:

```text
20e5cdd90d0601d71e9a011006cce92a7cde44709df95a74cc82ffaee30946a7  pass1_eigvals_golden_before_localfallback.log
02898a73eb09bd2535ab75d65c8af02118fa1c304db98c3bc014df0d2cdc61e6  pass1_baseline_eigvals_256.log
70344231136647e959f180ecc6c6e44cd70060615305df0bfa9b75462cfaeca7  pass1_baseline_eig_256.log
```

## Profile-Backed Target

Recent landed work moved the dominant wall from Hessenberg reduction/eigenvector back-transform to the serial Francis QR tail. On the current `vmi1227854` baseline, full eig spends roughly `22.447 ms` over eigvals at n=256, while eigvals itself remains `27.445 ms`. qglh3 targets the eigvals/shared Francis-QR floor with AED, not another eigenvector-only lever.

## Isomorphism Status

No source code changed in Pass 1.

- Ordering/tie-breaking: unchanged.
- Floating-point evaluation: unchanged.
- RNG: none.
- Golden outputs: pre-edit digests captured above.
