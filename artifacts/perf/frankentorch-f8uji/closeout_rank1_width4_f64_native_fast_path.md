# frankentorch-f8uji closeout

Bead: `frankentorch-f8uji`

Lever: guarded rank-1 width-4 F64 native state-dict header fast path.

Source commit: `776bcc69` (`perf(ft-serialize): native fast path for rank-1 width-4 f64 state-dict load`)

## Profile-backed target

Criterion row:

`native_state_dict/decode_many_small_f64_1024x4`

Baseline on `vmi1293453`:

`[276.58 us 286.65 us 300.14 us]`

Candidate on `vmi1293453`:

`[270.75 us 276.63 us 281.97 us]`

Median speedup:

`286.65 / 276.63 = 1.036x` (`3.5%` faster)

Score:

`2.07 = Impact 1.036 * Confidence 4.0 / Effort 2.0`

Verdict: KEEP.

## Behavior proof

- Ordering: final output remains a `BTreeMap`; the fast path inserts identical owned keys and does not change final ordering.
- Duplicate keys: duplicate insertion declines the fast path and falls back to the generic parser, preserving the existing duplicate-key error behavior.
- Tie-breaking: no sort comparator or tie policy changed.
- Floating point: width-4 F64 payloads are reconstructed with exact little-endian `u64` bits and `f64::from_bits`; the raw-bit test passed.
- DType/shape: the fast path accepts only rank-1 shape `[4]` with `DType::F64`; all other shapes/dtypes decline to the generic parser.
- Malformed inputs: truncated width-4 payloads and non-canonical records decline to the generic parser; malformed-input native tests passed.
- RNG: not involved.
- Golden SHA: `sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing` passed for all present fixtures, including `ft_serialize_decode_pass19.txt`.

## Gates

- RCH `vmi1227854`: `cargo test -j 1 -p ft-serialize native_ -- --nocapture` passed `18/18`.
- RCH `vmi1156319`: `cargo check -j 1 -p ft-serialize --all-targets` passed.
- RCH `vmi1153651`: `cargo clippy -j 1 -p ft-serialize --all-targets -- -D warnings` passed.
- Local: `cargo fmt -p ft-serialize --check` passed.
- UBS: `ubs crates/ft-serialize/src/lib.rs` exited 0 with no critical findings.

## Notes

The source implementation predates this closeout artifact in commit `776bcc69`; this artifact records the missing same-worker candidate timing, isomorphism proof, quality gates, and keep score for the bead closeout.
