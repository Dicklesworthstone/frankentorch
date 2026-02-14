# Foundation Performance Rebaseline (2026-02-14)

Bead: `bd-3v0.8`

## Baseline Refresh

Command:
- `~/.local/bin/rch exec -- bash -lc "cd /data/projects/frankentorch && env CARGO_TARGET_DIR=/tmp/frankentorch-target-self /usr/bin/time -f 'max_rss_kb=%M elapsed_s=%e' cargo test -q -p ft-conformance microbench_produces_percentiles -- --nocapture"`

Measured output:
- `microbench_ns p50=3477 p95=4698 p99=4698 mean=7509`
- `max_rss_kb=44192 elapsed_s=0.14`

Interpretation:
- step-time tails captured via strict-mode scalar DAC microbench.
- backward overhead is included (`run_scalar_microbench` executes `backward` each iteration).
- memory churn baseline is tracked via peak RSS snapshot (`/usr/bin/time`).
- command execution is CPU-offloaded through `rch` as required by project policy.

## Optimization Loop Linkage

Retained optimization lever:
- packet-level parallel validation in `validate_phase2c_artifacts`
- evidence: `artifacts/optimization/2026-02-14_packet_parallel_validation.md`

Behavior-isomorphism proof anchors:
- `artifacts/optimization/2026-02-13_phase2c_isomorphism.md`
- `artifacts/phase2c/conformance/differential_report_v1.json`
- `artifacts/phase2c/e2e_forensics/e2e_matrix_full_v1.jsonl`

## Gate Anchors

Post-change gate set (historically green and retained as behavioral anchors):
- `cargo fmt --check`
- `cargo check --all-targets`
- `cargo clippy --all-targets -- -D warnings`
- `cargo test --workspace`
- `cargo test -p ft-conformance -- --nocapture`
- `cargo bench`
