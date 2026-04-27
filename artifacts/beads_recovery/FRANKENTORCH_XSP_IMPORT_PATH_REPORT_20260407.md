# frankentorch-xsp import-path investigation report

## Scope

This report captures the results of the `frankentorch-xsp` investigation into the recurring
`.beads/beads.db` integrity warning:

`PRAGMA integrity_check` -> `Page N: never used`

The question was which write path reintroduced the anomaly after a clean `VACUUM INTO` rewrite.

## Inputs

- clean baseline DB:
  - `.beads/beads.db.vacuum_c84_20260407T013244Z`
- current tracker JSONL:
  - `.beads/issues.jsonl`
- archived probe artifacts:
  - `artifacts/beads_recovery/xsp_probe_20260407T025948Z/results.json`
  - `artifacts/beads_recovery/xsp_import_focus_20260407T030054Z/results.json`
  - `artifacts/beads_recovery/xsp_sequence_20260407T030136Z/results.json`
  - `artifacts/beads_recovery/xsp_repeat_import_20260407T030112Z/results.json`

## Key findings

1. The clean vacuumed DB is genuinely clean.
   - baseline page count: `89`
   - baseline freelist count: `0`
   - `integrity_check`: `ok`

2. The corruption is not caused by generic read-only access or the `br doctor` write probe.
   - preserved integrity:
     - `br info --json`
     - `br doctor`
     - `br doctor --no-auto-flush --no-auto-import`
     - `br sync --status --json`
     - `br list --json --no-auto-import`
     - `br ready --json --no-auto-import`

3. The corruption is reintroduced by the import path.
   - explicit trigger:
     - `br sync --import-only --json`
   - implicit triggers through auto-import:
     - `br list --json`
     - `br ready --json`
     - `br show frankentorch-xsp --json`
     - `br update frankentorch-xsp --status ... --json`

4. The shared signature is `blocked_issues_cache` object churn.
   - clean baseline blocked-cache objects:
     - table rootpage `19`
     - PK autoindex rootpage `20`
     - `idx_blocked_cache_blocked_at` rootpage `56`
   - after import:
     - table rootpage `96`
     - PK autoindex rootpage `95`
     - `idx_blocked_cache_blocked_at` rootpage `97`
   - post-import integrity warning:
     - `Page 19: never used`
     - `Page 20: never used`
     - `Page 56: never used`

5. Inference:
   - The import path rebuilds or recreates `blocked_issues_cache`.
   - The new table/index objects land on fresh pages.
   - The old blocked-cache pages are not returned to the freelist correctly.
   - This leaves orphaned pages that `integrity_check` reports as `never used`.

## 2026-04-27 z2w revalidation

`frankentorch-z2w` repeated the recovery probe after `Dicklesworthstone/beads_rust#224` was
closed. The installed tool was:

- `br version 0.1.38 (release) (v0.1.38@8c5b245)`
- upstream #224 fix commit cited in the issue: `08a1a90`
- GitHub compare result: local `8c5b245` is 8 commits ahead of `08a1a90`

Two inputs were tested from the clean vacuumed DB
`.beads/beads.db.vacuum_c84_20260407T013244Z`:

1. Archived xsp JSONL fixture:
   `artifacts/beads_recovery/c84_snapshot_20260407T013033Z/issues.jsonl`
2. Current repo JSONL:
   `.beads/issues.jsonl`

Results:

- archived fixture: `br sync --import-only --json`, plain `br list --json`, plain
  `br ready --json`, and plain `br show frankentorch-xsp --json` all preserved
  `PRAGMA integrity_check = ok`; page count stayed `89`, freelist count stayed `0`
- current JSONL: `br info --json`, `br doctor`, `br sync --status --json`,
  `br list --json --no-auto-import`, and `br ready --json --no-auto-import` preserved
  `PRAGMA integrity_check = ok`; page count stayed `89`, freelist count stayed `0`
- current JSONL import or auto-import paths still failed the orthogonal SQLite
  `PRAGMA integrity_check`:
  - `br sync --import-only --json`
  - plain `br list --json`
  - plain `br ready --json`
  - plain `br show frankentorch-z2w --json`
  - plain `br update frankentorch-z2w --status in_progress --json`
- the current-JSONL failure signature changed from orphaned old rootpages to:
  - page count `89 -> 109`
  - freelist count `0 -> 4`
  - `integrity_check`: `Tree 12 page 12: free space corruption`
  - `blocked_issues_cache` and `idx_blocked_cache_blocked_at` rootpages stayed at `19` and `56`

Interpretation:

- The original #224 `blocked_issues_cache` drop/create rootpage churn appears fixed for the
  archived xsp fixture.
- Importing the current repo JSONL into the same clean vacuumed DB still creates an integrity
  warning visible to the orthogonal SQLite checker, even though `br doctor` reports its internal
  `sqlite.integrity_check` as OK.
- The local `--no-auto-import` recovery rule remains necessary for freshly vacuumed recovery DBs
  until this residual current-JSONL import corruption is understood.

Revalidation artifacts were generated under:

- `artifacts/beads_recovery/z2w_revalidation_20260427T204133Z/results.json`
- `artifacts/beads_recovery/z2w_revalidation_20260427T204133Z/archived_jsonl_results.json`

## Reproduction summary

### Probe matrix

Starting from the clean vacuumed DB plus current JSONL:

- safe:
  - `br info --json`
  - `br doctor`
  - `br sync --status --json`
- unsafe:
  - `br sync --import-only --json`
  - any command that auto-imports before operating

### Minimal reproducer

1. Start from `.beads/beads.db.vacuum_c84_20260407T013244Z`
2. Pair it with the current `.beads/issues.jsonl`
3. Run `br sync --import-only --json`
4. Run `PRAGMA integrity_check`

Observed result:

- page count grows from `89` to `97`
- freelist count becomes `4`
- integrity check reports:
  - `Page 19: never used`
  - `Page 20: never used`
  - `Page 56: never used`

## Operational guidance

Until the residual current-JSONL import corruption is fixed:

- treat the import path as unsafe for freshly vacuumed recovery DBs
- use `--no-auto-import` when inspecting a known-good vacuumed recovery DB
- prefer `br doctor` or `br info` for non-mutating diagnostics on recovered DBs
- assume any command that auto-imports can reintroduce an integrity warning

## Local recovery playbook

Use this when a tracker recovery session starts from a freshly vacuumed `.beads/beads.db`.

### Safe inspection commands

- `br info --json`
- `br doctor`
- `br sync --status --json`
- `br list --json --no-auto-import`
- `br ready --json --no-auto-import`

### Commands to avoid on the vacuumed recovery DB

- `br sync --import-only --json`
- plain `br list --json`
- plain `br ready --json`
- plain `br show <id> --json`
- plain `br update <id> --status ... --json`
- any other command path that auto-imports before operating

### Recovery-session rule

If you need to inspect a known-good vacuumed recovery DB before the upstream fix lands, stay on
`--no-auto-import` variants and treat the import path as destructive to structural cleanliness even
when logical issue data remains usable.

## Upstream escalation

The original issue was escalated upstream in `beads_rust`:

- `Dicklesworthstone/beads_rust#224`
- https://github.com/Dicklesworthstone/beads_rust/issues/224

The upstream report includes:

- the minimal reproducer from the clean vacuumed baseline
- before/after rootpage movement for `blocked_issues_cache`
- exact `integrity_check` failure output
- the current local workaround (`--no-auto-import` on recovered DBs)

On 2026-04-27, `frankentorch-z2w` added fresh upstream evidence: the archived reproducer passes
with `br 0.1.38`, but importing the current FrankenTorch JSONL still leaves orthogonal SQLite
`PRAGMA integrity_check` reporting `Tree 12 page 12: free space corruption`.

## Remaining work

This investigation isolated the original failing class of operations and the z2w pass narrowed the
residual failure to current-JSONL import behavior. Remaining follow-up should track:

- determining why the current FrankenTorch JSONL import still produces orthogonal SQLite
  free-space corruption while the archived xsp JSONL passes
- removing the temporary `--no-auto-import` recovery restriction only after both archived and
  current-JSONL recovery imports preserve orthogonal `PRAGMA integrity_check = ok`
