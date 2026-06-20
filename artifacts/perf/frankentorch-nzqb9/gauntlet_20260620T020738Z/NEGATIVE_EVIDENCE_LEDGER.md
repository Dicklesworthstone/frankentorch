# frankentorch-nzqb9 Negative Evidence Ledger

Agent: IvoryDeer / cod-b
Date: 2026-06-20

## Rejected Or Weak Attempts

| Attempt | Environment | Evidence | Verdict |
| --- | --- | --- | --- |
| Materialize-forward fused sum-loss draft | `vmi1153651`, abandoned dirty scratch worktree | materialized 17.519 ms, fused 18.345 ms, 0.955x | Rejected. Not committed. |
| v1 flat-index fused path, local PyTorch ratio | local PyTorch venv | materialized 6.9760 ms, fused 7.0877 ms, PyTorch 1.5325 ms; ratio worsened from 4.552x slower to 4.625x slower | Rejected as final form. Replaced with row-major leaf walk and lower fused-pool parallel threshold. |
| v1 flat-index fused path, remote all-row attempt | `rch` selected `vmi1149989` despite `RCH_WORKER=hz2` | materialized 3.9581 ms, fused 5.8278 ms, 0.679x; PyTorch row failed with `ModuleNotFoundError: No module named 'torch'` | Negative routing evidence. Not used as same-worker PyTorch proof. |
| Final v2 fused path vs PyTorch | local PyTorch venv | fused 5.0290 ms, PyTorch 1.9905 ms, 2.527x slower | Strict PyTorch head-to-head remains a loss. Kept only because FT ratio improved from 3.280x slower to 2.527x slower and correctness gates are green. |
| `cargo fmt --check` | workspace | red across broad pre-existing formatting diffs in unrelated regions | Not fixed in this perf commit to avoid unrelated workspace churn. Overlapping new hunks were manually formatted. |
| UBS scan | touched files; `ubs_changed_files_final_20260620T020738Z.log` | broad pre-existing findings across large files: 1001 critical, 34433 warning, 5288 info; embedded clippy/check probes green and no targeted cargo check/clippy/conformance failure from the fused path | Not used as a blocker for this perf lever; scanner debt should be handled separately. |

## Win/Loss Accounting

Strict "does FrankenTorch beat PyTorch?" tally for final ratio proof:

- Wins: 0
- Losses: 1
- Neutral: 0

Lever-vs-current-FrankenTorch tally across measured Rust comparisons:

- Wins: 4 (`hz2` v1, `hz2` v1 confirm, `hz2` v2, local v2)
- Losses: 3 (materialize-forward draft, local v1, remote `vmi1149989` v1)
- Neutral: 0

The kept result is therefore a measured gap reduction, not a PyTorch domination claim.
