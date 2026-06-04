# ft-dispatch schema registry digest bucket index

Bead: frankentorch-e6hq
Date: 2026-06-04
Agent: codex

## Target

Profile-backed target: `schema_registry/register_1024` in `crates/ft-dispatch/benches/dispatch_bench.rs`.

Ready queue was empty. Existing in-progress perf beads were owned by other agents for `ft-kernel-cpu` and `ft-optim`, so this pass continued on the already reprofiled `ft-dispatch` schema-registry hotspot.

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-dispatch --bench dispatch_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts2`

Result:

```text
schema_registry/register_1024 time: [293.61 us 295.77 us 297.38 us]
```

## Lever

One lever: replace the registry's duplicate `HashMap<String, usize>` index with a digest-keyed map:

```text
HashMap<u64, SchemaIndexBucket, SchemaNameBuildHasher>
```

The bucket is `Single(usize)` for the common case and expands to `Collision(Vec<usize>)` only when two normalized names share the same digest. Lookups and duplicate checks still compare the full `normalized_name` before accepting a match.

This removes the second normalized-name allocation in the map key and avoids variable-length string-key hashing on every registration while preserving collision correctness.

## Isomorphism Proof

- Ordering: `entries` push order is unchanged; `iter()` still sorts by `normalized_name`.
- Duplicate behavior: duplicate detection still runs before `BinaryOp::from_schema_base`, preserving duplicate-before-unsupported diagnostics.
- Collision behavior: internal forced-collision unit test proves the bucket checks the full normalized name and rejects unrelated names.
- Floating point: no floating-point arithmetic is touched.
- RNG/ties: no RNG, tie-breaking, dispatch priority, or kernel selection path is touched.
- Public schema entries: `SchemaDispatchEntry` fields and returned normalized names are unchanged.
- Golden outputs: existing schema-registry golden tests passed, and fixture sha256 values remained unchanged.

Golden sha256:

```text
683125fea979d9285c658d2c794eb97380d551bd14813291f14fb2b1a94d3dae  artifacts/optimization/golden_outputs/ft_dispatch_schema_registry_normalized_once_frankentorch-kgs4-15.txt
1ed775b109f682bad9b11c04e7818845bd57927ad48ef497c7ee2d6b7049207d  artifacts/optimization/golden_outputs/ft_dispatch_schema_pass21.txt
```

## Proof Commands

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-dispatch schema_registry -- --nocapture
cargo fmt -p ft-dispatch -- --check
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-dispatch --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-dispatch --all-targets --no-deps -- -D warnings
sha256sum artifacts/optimization/golden_outputs/ft_dispatch_schema_registry_normalized_once_frankentorch-kgs4-15.txt artifacts/optimization/golden_outputs/ft_dispatch_schema_pass21.txt
```

Result:

```text
cargo test: 10 passed; 0 failed; 98 filtered out
cargo fmt --check: passed
cargo check: passed
cargo clippy: passed
sha256sum: unchanged values listed above
```

Note: `ft-kernel-cpu` emitted a pre-existing duplicate `#[must_use]` warning from peer-owned work during crate-scoped builds; `ft-dispatch` passed its gates.

## Rebench

Command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-dispatch --bench dispatch_bench -- --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts2`

Result:

```text
schema_registry/register_1024 time: [195.00 us 195.49 us 195.99 us]
```

Delta:

```text
median 295.77 us -> 195.49 us
speedup 1.51x
improvement 33.9%
```

Score:

```text
Impact 3 x Confidence 5 / Effort 2 = 7.5
```

Decision: keep.
