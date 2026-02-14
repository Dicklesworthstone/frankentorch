# 2026-02-14 Packet Parallel Validation Optimization

Optimization lever: packet-level parallelism in `validate_phase2c_artifacts` with deterministic sort retained.

## Hotspot Context

- Target command:
  - `/tmp/frankentorch-target/debug/validate_phase2c_artifacts /tmp/ft_phase2c_bench_before_ep1sp0 >/dev/null`
- Workload:
  - synthetic large corpus root with many packet directories (`/tmp/ft_phase2c_bench_before_ep1sp0`)

## Baseline vs Optimized (Same Revision Toggle)

- Baseline (parallelism disabled):
  - `FT_DISABLE_PACKET_PARALLELISM=1 /tmp/frankentorch-target/debug/validate_phase2c_artifacts /tmp/ft_phase2c_bench_before_ep1sp0 >/dev/null`
  - mean `66.2 ms` (25 runs, warmup 3)
- Optimized (parallelism enabled):
  - `/tmp/frankentorch-target/debug/validate_phase2c_artifacts /tmp/ft_phase2c_bench_before_ep1sp0 >/dev/null`
  - mean `35.3 ms` (25 runs, warmup 3)
- Delta:
  - `1.88x` faster with parallel packet validation

## Isomorphism Proof

Change: execute `validate_packet` across worker chunks, then sort by `packet_id` before summary emission.

- Ordering preserved:
  - Yes. Final `packets` vector is sorted lexicographically by `packet_id` exactly as before output serialization.
- Tie-breaking unchanged:
  - Yes. Duplicate packet IDs are not expected; stable ordering by packet ID is deterministic.
- Floating-point:
  - N/A (no floating-point computation in validator logic path).
- RNG seeds:
  - N/A (no RNG used).
- Golden behavior checks:
  - Validator test suite remains green (`cargo test -p ft-conformance --bin validate_phase2c_artifacts`).
  - Full workspace/tests/clippy gates remain green post-change.

## Risk Notes

- Added env override:
  - `FT_DISABLE_PACKET_PARALLELISM=1` for deterministic single-thread fallback and forensic A/B comparison.
- Failure mode:
  - Worker panic escalates immediately (`expect`) to avoid silent partial validation.
