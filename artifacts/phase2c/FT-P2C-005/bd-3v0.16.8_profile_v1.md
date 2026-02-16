# FT-P2C-005 Profiling Note — bd-3v0.16.8

Generated: `2026-02-16T04:08:07Z`

## Objective

Measure the FT-P2C-005 packet e2e latency impact of preloading scalar/tensor-meta/dispatch fixtures and reusing them across strict+hardened mode execution.

## Measurement Command

```bash
rch exec -- cargo test -p ft-conformance packet_e2e_microbench_cpu_kernel_legacy_vs_optimized_profiles -- --nocapture
```

The comparator test executes both legacy and optimized paths in the same process and worker allocation to minimize cross-worker timing skew.

## Raw Comparator Output

```text
packet_e2e_microbench_compare_ns packet=FT-P2C-005 legacy_p50=10282230 legacy_p95=14150815 legacy_p99=14150815 legacy_mean=10814785 optimized_p50=9629848 optimized_p95=9885917 optimized_p99=9885917 optimized_mean=9232517
```

## Delta Summary

- `p50`: `-652382ns` (`6.345%` faster)
- `p95`: `-4264898ns` (`30.139%` faster)
- `p99`: `-4264898ns` (`30.139%` faster)
- `mean`: `-1582268ns` (`14.631%` faster)

## Isomorphism Guardrails

- Projection namespace and envelope regression checks remain green:
  - `e2e_matrix_packet_filter_includes_cpu_kernel_packet_entries`
  - `e2e_matrix_unfiltered_keeps_ft_p2c_005_projection_entries`
- Differential and replay artifacts are unchanged and still linked:
  - `artifacts/phase2c/FT-P2C-005/differential_packet_report_v1.json`
  - `artifacts/phase2c/e2e_forensics/ft-p2c-005.jsonl`

## Memory/Tooling Note

This microbench reports latency tails only; RSS deltas are not emitted by this harness and remain tracked via dedicated perf harness integration.
