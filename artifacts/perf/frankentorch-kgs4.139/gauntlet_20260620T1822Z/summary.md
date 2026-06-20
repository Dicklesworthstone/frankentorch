# frankentorch-kgs4.139 - BatchNorm1d automatic tensor_sum shortcut

Verdict: kept. The ordinary f64 training-mode
`functional_batch_norm1d(...).tensor_sum()` path now routes Sum backward through
the BatchNorm scalar-loss backward helper when the BatchNorm output has no
retained grad or hooks. Observable output-gradient cases fall back to the
materialized Sum path.

Primary A/B:

- Baseline local fallback from corrected RCH command:
  - native ordinary path: `11.622 ms`
  - explicit scalar-sum API: `5.0014 ms`
  - fold-reference: `59.337 ms`
- Candidate local same-machine:
  - native automatic shortcut: `6.6151 ms`
  - explicit scalar-sum API: `5.1754 ms`
  - fold-reference: `40.052 ms`
- Internal ratio: `6.6151 / 11.622 = 0.5692x`, or `1.76x` faster.
- Automatic shortcut remains `1.278x` slower than the explicit scalar API
  because it still materializes the BatchNorm output in forward.

PyTorch comparator:

- Local PyTorch `2.12.1+cpu`, 32 compute/inter-op threads, same NCL f64
  fixture with clone/detach per rep: `0.891630 ms` median.
- Local automatic shortcut/PyTorch ratio: `7.42x` slower.
- RCH after row on `hz2`: native automatic `6.0836 ms`, explicit scalar
  `4.7261 ms`, fold `48.006 ms`; since the before RCH row fell back locally,
  this is routing evidence only.

Gates:

- `rch exec -- cargo test -p ft-api functional_batch_norm1d_tensor_sum --lib -- --nocapture`: passed, 2/0.
- `rch exec -- cargo test -p ft-api functional_batch_norm1d --lib -- --nocapture`: passed, 10/0.
- `rch exec -- cargo test -p ft-conformance`: passed.
- `rch exec -- cargo check -p ft-autograd --lib`: passed.
- `rch exec -- cargo check -p ft-api --lib --benches`: passed after formatting patch.
- `rch exec -- cargo clippy -p ft-autograd --lib -- -D warnings`: passed.
- `rch exec -- cargo clippy -p ft-api --lib -- -D warnings`: passed.
- `rch exec -- cargo clippy -p ft-api --lib --benches -- -D warnings`: failed on pre-existing ft-api test lint debt outside this change.
- `git diff --check`: passed.
- Full-file rustfmt remains blocked by pre-existing drift; touched shortcut hunks were manually formatted and the touched-symbol rustfmt grep is clean.
- Scoped UBS on Rust source/docs/summary timed out after 240s while scanning the large Rust files, with no findings emitted first. Docs/artifact-only UBS exited 0 but reported Markdown as no recognizable language.
