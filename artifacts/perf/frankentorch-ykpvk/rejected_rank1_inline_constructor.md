# frankentorch-ykpvk rejection

Bead: `frankentorch-ykpvk`

Candidate: checked-by-construction `DenseTensor::from_rank1_f64_inline4_cpu` in `ft-core`, called only by the canonical width-4 F64 native decode fast path.

## Profile-backed target

Post-f8uji RCH reprofile on `vmi1293453`:

`native_state_dict/decode_many_small_f64_1024x4`

Baseline:

`[254.89 us 260.78 us 270.87 us]`

Candidate probe:

`[263.65 us 289.49 us 316.91 us]`

The raw captured bench artifact selected `vmi1227854`, not the baseline
worker `vmi1293453`; treat the candidate row as rejection/routing evidence,
not as a decisive same-worker regression proof.

Cross-worker recorded median ratio:

`260.78 / 289.49 = 0.901x`

Verdict: REJECT. Score `0.0`; the source hunk was removed.

## Behavior proof

- RCH `vmi1149989`: focused `ft-core` constructor test passed `1/1`.
- RCH `vmi1227854`: `cargo test -j 1 -p ft-serialize native_ -- --nocapture` passed `18/18`.
- Golden SHA-256 verification passed for all present fixtures, including `ft_serialize_decode_pass19.txt`.
- RCH `vmi1153651`: `cargo check -j 1 -p ft-core -p ft-serialize --all-targets` passed.
- RCH `vmi1153651`: `cargo clippy -j 1 -p ft-core -p ft-serialize --all-targets -- -D warnings` passed.
- Local `cargo fmt -p ft-core -p ft-serialize --check` passed.
- UBS on `crates/ft-core/src/lib.rs` and `crates/ft-serialize/src/lib.rs` exited 0 with no critical findings.

## Source state

The candidate source hunk was removed after the regression. Final source has no ykpvk diff.

## Reroute

Do not retry a generic constructor-bypass family for this decode path. The next ft-serialize attempt should use a deeper parser/storage-layout primitive, such as batch-owned tensor metadata construction or a fixed-layout state-dict container that still materializes a `BTreeMap` with identical observable ordering.
