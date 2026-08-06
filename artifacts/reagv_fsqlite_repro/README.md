# frankentorch-reagv — minimal reproducer for the fsqlite false-positive integrity finding

## Where the reproducer file lives — read this first

The 544 KB `minimal_repro.db` itself is **not in git**: `.gitignore:133` ignores
`artifacts/beads_recovery/`, which is where recovery artifacts belong, and that
rule is deliberate. The file sits at
`artifacts/beads_recovery/reagv_repro/minimal_repro.db` on the box that produced
it. Attach it to the upstream issue out of band, or rebuild it with the recipe in
[Provenance](#provenance) — the reduction is deterministic from any live tracker
DB exhibiting the finding.

This document is the part worth versioning: the contrast, and the eight
hypotheses already eliminated.

## What this is

`minimal_repro.db` (544 KB) is a beads tracker database, reduced to **1 issue and
1 event**, on which:

| checker | verdict |
|---|---|
| canonical `sqlite3` `PRAGMA integrity_check` | **ok** |
| canonical `sqlite3` `PRAGMA foreign_key_check` | one unrelated `child_counters` row; no `events` finding |
| `br 0.2.20` (`br doctor`, fsqlite-backed) | **ERROR: database disk image is malformed: index `idx_events_type` entry for rowid 1 does not match the table row payload** |

Reproduce:

```
cd <a scratch dir>; mkdir -p .beads
cp minimal_repro.db .beads/beads.db
cp <any beads workspace>/.beads/config.yaml <any beads workspace>/.beads/metadata.json .beads/
br doctor          # fsqlite reports the malformed index
sqlite3 .beads/beads.db 'PRAGMA integrity_check;'   # prints: ok
```

## Why it matters

This finding is what deadlocks `br 0.2.20` on this repo's tracker:
`doctor migrate-schema` refuses because of the integrity finding, and
`doctor --repair-indexes` refuses because of the schema-version mismatch that
migrate-schema exists to fix. Neither escape hatch is reachable, so the whole
fleet is pinned to `br 0.1.27`.

## What was ruled out

Each hypothesis was tested and **refuted**, so an upstream fix does not need to
re-walk them:

| hypothesis | test | result |
|---|---|---|
| Real corruption | `sqlite3 PRAGMA integrity_check` after REINDEX + full index rebuild + `VACUUM` | **ok** — the file is valid by canonical SQLite |
| Overflow pages from multi-KB comments | kept only rows `>= 4000 B` payload (4 rows) | **OK** — large rows alone do not trigger it |
| ...and the converse | kept only rows `< 500 B` payload (6141 rows) | **ERROR** — small rows do trigger it |
| Row count / B-tree depth | bisected 6141 → 3000 → 1000 → 300 → 100 → 30 → 15 → 8 → 4 → 3 → 2 → **1** | **ERROR at 1 row**; not a volume effect |
| Key cardinality | a fresh DB with 8000 synthetic events (1 distinct `issue_id`) | **OK** |
| Schema version | `PRAGMA user_version` bumped 16 → 17 on a copy | **ERROR persists** — not the version gate |
| `sqlite_master` table/index adjacency | fresh DB: added a table, then an index on `events`, so an unrelated table is the last-seen entry before it | **OK** — not a positional/last-seen-table scan bug |
| The row's own content | inserted the byte-identical row into a fresh `br init` DB | **OK** |

## Where that leaves it

The trigger is a property of **the file**, not of the row content, the row count,
the schema text, the schema version, the index/table ordering, or overflow pages.
The same single row is fine in a fresh file and reports malformed in this one, and
canonical SQLite reads this file as valid.

So this is an **fsqlite reader defect on a database that SQLite considers
healthy** — the remaining work is inside fsqlite's index-vs-row validation, which
is why this reproducer exists rather than a further FrankenTorch-side bisect.

A second detector on the same read path is wrong in the same direction on larger
copies of this file: `db.null_defaults` reports `events.created_at has 2202 NULL
value(s)` where `sqlite3` counts **0** for both `IS NULL` and
`typeof(created_at)='null'`. That miscount does scale with row count (a 1-row copy
reports `OK`), so it is a second symptom of the same misread rather than a
constant.

## Provenance

Derived from `/data/projects/frankentorch/.beads/beads.db` on 2026-08-06, after
that database's **genuine** index corruption had already been repaired
(`frankentorch-10znp`) and verified `ok` by canonical SQLite. Reduced by deleting
`events` rows with `id > 1`, then all `issues` except the one the surviving event
references, then `comments`/`dependencies`/`labels`, then `VACUUM`.
