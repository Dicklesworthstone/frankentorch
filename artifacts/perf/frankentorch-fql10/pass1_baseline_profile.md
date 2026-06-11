# frankentorch-fql10 pass 1 baseline/profile

Date: 2026-06-11
Agent: BlackThrush
Git HEAD: `446652100c5165304eab22ca42fa5c681e0babef`

## Scope

No source edits. This pass only captured baseline/profile evidence for
`frankentorch-fql10` under `artifacts/perf/frankentorch-fql10/`.

## RCH admission

Remote-only attempts failed before execution:

- `vmi1227854` pinned Criterion baseline:
  `pass1_baseline_eig_eigvals_256_vmi1227854.log`
- unpinned remote Criterion baseline:
  `pass1_baseline_eig_eigvals_256_remote.log`
- `vmi1152480` pinned Criterion baseline:
  `pass1_baseline_eig_eigvals_256_vmi1152480.log`
- `vmi1227854` pinned golden:
  `pass1_eigvals_golden_remote_attempt.log`

All failures reported:
`[RCH] local (no admissible workers: critical_pressure=1,active_project_exclusion=1)`
followed by remote-required refusal where `RCH_REQUIRE_REMOTE=1` was set.

The successful measurements used `rch exec` local fallback on `thinkstation1`.
Both Criterion rows ran in one command, so they are same-host/same-process
fallback evidence, not remote same-worker proof.

## Criterion baseline

Command:

```bash
rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- '^(eigvals_f64_256x256|eig_f64_256x256)$' --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Log: `pass1_baseline_eig_eigvals_256_localfallback.log`

| Row | Estimate |
| --- | ---: |
| `eig_f64_256x256` | `[42.391 ms 43.219 ms 43.875 ms]` |
| `eigvals_f64_256x256` | `[27.081 ms 27.472 ms 27.835 ms]` |

## Golden output

Command:

```bash
rch exec -- cargo run -p ft-kernel-cpu --release --example eigvals_golden
```

Artifacts:

- stdout: `pass1_eigvals_golden_stdout.txt`
- stderr/RCH log: `pass1_eigvals_golden_localfallback.log`
- sha256 manifest: `pass1_eigvals_golden_stdout.sha256`

Golden stdout digest:

```text
24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725  artifacts/perf/frankentorch-fql10/pass1_eigvals_golden_stdout.txt
```

Fixture digests printed by the example:

| n | `eigvals_digest` | `eig_digest` |
| ---: | --- | --- |
| 64 | `0xbc0583d464b1a211` | `0xbc0583d464b1a211` |
| 128 | `0x763c4b15d92c4b89` | `0x763c4b15d92c4b89` |
| 256 | `0x00b87b4996340204` | `0x00b87b4996340204` |

`sha256sum -c pass1_eigvals_golden_stdout.sha256` passed.

## Profile target

The next real target is the serial Francis QR residual in
`eig_francis_schur`: the shared QR floor for both `eigvals` and `eig`.
Existing qglh3 evidence already rules out adjacent micro-levers:

- not eigenvector back-substitution;
- not `q_acc` replay/whole-stream machinery;
- not more values-only AED threshold or whole-window suffix variants.

The route is full AED window-record plumbing and/or small-bulge multishift QR
with residual/orthogonality/order proof, not another threshold-only shortcut.
