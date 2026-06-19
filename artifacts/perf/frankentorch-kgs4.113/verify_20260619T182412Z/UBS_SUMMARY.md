# UBS summary

Command:

```bash
ubs crates/ft-api/benches/pytorch_gauntlet_bench.rs \
  crates/ft-api/benches/pytorch_sdpa_grad.py \
  docs/NEGATIVE_EVIDENCE.md \
  docs/RELEASE_READINESS_SCORECARD.md \
  artifacts/perf/frankentorch-kgs4.113/verify_20260619T182412Z/SCORECARD.md \
  artifacts/perf/frankentorch-kgs4.113/verify_20260619T182412Z/NEGATIVE_EVIDENCE_LEDGER.md \
  artifacts/perf/frankentorch-kgs4/sdpa_scaled_gemm_alpha_code_first.md
```

Result:

- Python scan: 0 critical, 0 warning.
- Rust scan: 0 critical, 0 warning.
- Combined summary: 2 source files scanned, 0 critical, 0 warning, 29 info.

The raw terminal transcript was not committed because the UBS banner contains
decorative trailing whitespace; this summary preserves the actionable result.
