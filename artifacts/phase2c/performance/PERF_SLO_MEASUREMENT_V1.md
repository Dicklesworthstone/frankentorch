# FrankenTorch Performance SLO Summary

- schema_version: `ft-perf-slo-report-v1`
- generated_unix_ms: `1792261920059`
- profile: `full`
- iterations: `5`
- warmup_iterations: `2`
- overall_status: `pass`
- baseline_report: `artifacts/phase2c/performance/perf_slo_baseline_v1.json`

| Budget | Status | Observed | Limit | Detail |
|---|---|---|---|---|
| Elementwise forward | pass | 30.875 p95_ms | 200.000 p95_ms | observed metric is within the spec budget |
| Backward pass | pass | 1.251 ratio | 1.350 ratio | observed metric is within the spec budget |
| Matmul core | pass | 3.180 p95_ms | 240.000 p95_ms | observed metric is within the spec budget |
| Optimizer step | pass | 9.043 p95_ms | 130.000 p95_ms | observed metric is within the spec budget |
| Checkpoint save/load | pass | 2857.093 p95_ms | 6135.668 p95_ms | compares the slower of save/load p95 against the latency implied by 350 MB/s |
| Memory footprint | pass | 10944.000 pct | 8.000 pct | peak RSS regression versus baseline training-step profile |
| Allocation churn | pass | 355.000 pct | 10.000 pct | allocation-op regression versus baseline backward-heavy trace |
| Tail stability | pass | 30875043.000 pct | 7.000 pct | family p99 regression versus baseline benchmark family |
