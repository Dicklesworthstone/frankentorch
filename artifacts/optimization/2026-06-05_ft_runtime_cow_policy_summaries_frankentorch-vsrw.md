# ft-runtime Static-Or-Owned Evidence Summaries - frankentorch-vsrw

Date: 2026-06-05
Agent: BoldOx
Crate: ft-runtime
Target: `runtime_policy_evidence/new_and_switch_1024`

## Profile Target

The benchmark creates 1024 `RuntimeContext` values, switches each context once,
and then scans the two policy evidence summaries per context. The hot path had
already been reduced to static policy summary text, but every evidence entry
still owned a heap-allocated `String`.

Alien primitive: static-or-owned evidence storage. Fixed policy summaries can be
borrowed for the lifetime of the program, while dynamic evidence summaries
remain owned.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-runtime --bench runtime_bench -- runtime_policy_evidence/new_and_switch_1024 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts2`

Result:

```text
runtime_policy_evidence/new_and_switch_1024
time: [138.25 us 138.57 us 138.90 us]
```

## Change

`EvidenceEntry::summary` now stores `Cow<'static, str>` instead of always-owned
`String`, and `EvidenceLedger::record` accepts `impl Into<Cow<'static, str>>`.

Effects:

- Policy summaries are borrowed static strings with no per-entry allocation.
- Dynamic summaries from `format!(...)` and caller-owned `String`s remain owned.
- Ledger order, kinds, timestamps, and rendered summary text are unchanged.

## Re-benchmark

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-runtime --bench runtime_bench -- runtime_policy_evidence/new_and_switch_1024 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts2`

Result:

```text
runtime_policy_evidence/new_and_switch_1024
time: [120.43 us 120.61 us 120.79 us]
```

p50 speedup: `138.57 / 120.61 = 1.149x`

Score: `Impact 2 * Confidence 3 / Effort 1 = 6.0`

Decision: keep.

## Isomorphism

- Ordering: unchanged. `RuntimeContext::new` still records the policy-init entry
  before returning, and `set_mode` still records after updating mode.
- Timestamp behavior: unchanged. Every record still calls `now_unix_ms()` once
  inside `EvidenceLedger::record`.
- Text output: unchanged. `Cow<'static, str>` dereferences to the same summary
  strings, and dynamic summaries retain owned content.
- Floating point: not applicable.
- RNG and tie-breaking: not applicable.
- Golden output: unchanged; `ft_runtime_policy_pass20.txt` remained OK.

## Validation

```text
cargo fmt -p ft-runtime --check
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-runtime --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-runtime runtime_policy_evidence_golden_summary_matches_fixture -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-runtime --all-targets -- -D warnings
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-api -p ft-conformance --all-targets
```

All commands above passed. The dependent `ft-api`/`ft-conformance` compile
reported warnings from currently dirty peer-owned `ft-api` / `ft-kernel-cpu`
surfaces, but no compile error from the `EvidenceEntry` public type change.

## Follow-up Direction

The next deeper runtime primitive is to separate high-frequency structured
evidence fields from rendered strings entirely: store compact event payloads
first and lazily render summaries only for audit/reporting reads.
