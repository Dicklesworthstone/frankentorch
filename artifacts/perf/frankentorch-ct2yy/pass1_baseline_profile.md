# frankentorch-ct2yy pass 1 baseline/profile

Date: 2026-06-11
Agent: BlackThrush
Git HEAD: `446652100c5165304eab22ca42fa5c681e0babef`

## Scope

No source edits. This pass captured QR baseline evidence for
`frankentorch-ct2yy` under `artifacts/perf/frankentorch-ct2yy/`.

## RCH admission

Remote-required Criterion failed before execution:

- `pass1_qr_baseline_remote.log`
- refusal: `[RCH] local (no admissible workers: critical_pressure=1,insufficient_slots=1)`

The successful measurements used plain `rch exec` local fallback. Treat these as
routing evidence only. A keep/reject decision still requires same-worker remote
before/after evidence.

## Criterion baseline

Command:

```bash
rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- '^(qr_f64_512x512|qr_f64_tall_2048x128)$' --sample-size 10 --warm-up-time 1 --measurement-time 3
```

Log: `pass1_qr_baseline_rch_fallback.log`

| Row | Estimate |
| --- | ---: |
| `qr_f64_512x512` | `[53.778 ms 54.462 ms 55.012 ms]` |
| `qr_f64_tall_2048x128` | `[28.818 ms 29.168 ms 29.767 ms]` |

## Profile target

The tracker note says NB=32 -> 64 regressed, so the visible QR residual is the
scalar Householder panel factorization inside the already-blocked compact-WY QR
path, not the GEMM trailing update. The next route must be recursive panel
factorization / CAQR-style panel restructuring, not another NB-size tweak.
