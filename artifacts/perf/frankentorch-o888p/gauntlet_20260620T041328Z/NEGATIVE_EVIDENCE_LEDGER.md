# frankentorch-o888p Negative Evidence Ledger

Agent: IvoryDeer / cod-b
Date: 2026-06-20

## Win / Loss / Neutral Accounting

Strict final ratio-vs-PyTorch:

- Wins: 0
- Losses: 1 (`frankentorch_fused_sum_loss` 6.6646 ms vs PyTorch 1.8889 ms, 3.528x slower)
- Neutral: 0

Lever-vs-current-FrankenTorch evidence:

- Wins: 4
  - Local fused scalar-loss row: 7.7994 ms -> 6.6646 ms, 1.170x.
  - Local `frankentorch_backward_only`: 6.4558 ms -> 5.2178 ms, 1.237x.
  - Local `kernel_forward_with_indices`: 817.61 us -> 721.37 us, 1.133x.
  - Local `kernel_backward_from_indices`: 1.7696 ms -> 1.5417 ms, 1.148x.
- Neutral: 2
  - Materialized public trace row: 8.2494 ms -> 7.9640 ms, Criterion says no significant change.
  - PyTorch row: 1.9145 ms -> 1.8889 ms, Criterion says no significant change.
- Losses: 1
  - Setup tensor probe moved 209.35 us -> 237.30 us in this run; outside the changed kernel path.

## Rejected / Weak / Non-Blocking Evidence

| Evidence | Environment | Result | Decision |
| --- | --- | --- | --- |
| Final fused path vs PyTorch | local PyTorch venv, same harness | 6.6646 ms vs 1.8889 ms, 3.528x slower | Strict loss. Kept only as a measured ratio improvement. |
| Materialized public trace | local same harness | 8.2494 ms -> 7.9640 ms, no significant change | Neutral. This lever only targets the explicit fused scalar-loss path. |
| Setup tensor stage | local same harness | 209.35 us -> 237.30 us | Negative/noisy side evidence, outside touched kernel path. |
| `cargo clippy -p ft-kernel-cpu --all-targets -- -D warnings` | rch `hz2` | red on pre-existing lint debt: example single-element loop, `items_after_test_module`, unrelated test range loop, identity op | Not fixed in this perf commit to avoid broad lint cleanup. |
| `cargo fmt --check` | workspace | exit 1 with broad formatting diffs across ft-api examples/benches and unrelated crates | Not applied; rustfmt would rewrite unrelated files in the shared campaign. |
| UBS touched file scan | local shadow workspace | zero criticals, 4572 warnings, 1527 info items | Warning inventory is pre-existing broad kernel debt; no new critical blocker from this lever. |

## Remaining Gap Route

The kept kernel trim does not dominate PyTorch. Remaining measured gap after this pass:

- FrankenTorch fused: 6.6646 ms
- PyTorch 2.12 CPU: 1.8889 ms
- Gap: 3.528x slower

Next bead: `frankentorch-kfdnn`.

Candidate levers for that bead:

- Automatic graph fusion for `max_pool3d -> sum -> backward`, so the public materialized trace reaches the scalar-loss path.
- Session/autograd arena or object reuse for this tight train-step benchmark.
- More compact sidecar representation than `Vec<f64>` offsets if it preserves the tape storage contract or introduces a typed sidecar without broad churn.
- Deeper 2x2x2 stride-2 scheduling only if same-machine PyTorch ratio improves.
