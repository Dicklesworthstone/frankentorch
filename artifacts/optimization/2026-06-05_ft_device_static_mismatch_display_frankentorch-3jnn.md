# ft-device Static DeviceError Mismatch Display

- Bead: `frankentorch-3jnn`
- Agent: `BoldOx`
- Crate: `ft-device`
- Target row: `device_guard/mismatch_display_65536`
- Lever: static mismatch-display message table
- Verdict: keep

## Profile-Backed Baseline

The ready queue had no ready perf beads. Active perf work owned the
`ft-kernel-cpu` no-gaps/GEMM surface and `ft-optim`, so this pass used the
unowned `ft-device` Criterion surface.

Baseline command:

```text
RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -p ft-device --bench device_bench -- device_guard --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Worker: `ts1`

Baseline rows:

```text
device_guard/ensure_tensor_device_match_65536: [39.309 us 40.003 us 40.760 us]
device_guard/ensure_same_device_match_65536:   [40.325 us 40.758 us 41.183 us]
device_guard/mismatch_display_65536:           [1.3416 ms 1.3571 ms 1.3738 ms]
```

The mismatch display path remained the dominant ft-device benchmark row.

## Change

`DeviceError::Mismatch` display now routes each `(expected, actual)` device pair
to a fixed static message:

```text
device mismatch: expected Cpu, got Cpu
device mismatch: expected Cpu, got Cuda
device mismatch: expected Cuda, got Cpu
device mismatch: expected Cuda, got Cuda
```

This replaces four formatter writes plus two device-name lookups with one
`write_str` call for the exact final string.

## Isomorphism Proof

- Display bytes: unchanged. `device_error_display` now checks all four static
  table entries exactly.
- Error values: unchanged. `DeviceError::Mismatch { expected, actual }` still
  carries the same enum values.
- Ordering/tie-breaking: not applicable.
- Floating point: not applicable.
- RNG: not applicable.
- Guard behavior: unchanged. `ensure_tensor_device` and `ensure_same_device`
  construction of `DeviceError::Mismatch` is untouched.
- Golden output: existing
  `artifacts/optimization/golden_outputs/ft_device_error_display_frankentorch-mafg.txt`
  remained unchanged and passed `sha256sum -c`.

Proof commands:

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-device device_error_display -- --nocapture
```

Proof results:

```text
golden_checksums: all tracked outputs OK
device_error_display: 1 passed on ts1
```

## Rebench

Candidate command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=ts1 rch exec -- cargo bench -p ft-device --bench device_bench -- device_guard/mismatch_display_65536 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Candidate result:

```text
worker: ts1
device_guard/mismatch_display_65536: [526.45 us 530.72 us 534.68 us]
```

Delta:

```text
median: 1.3571 ms -> 530.72 us
speedup: 2.56x
elapsed: 60.9% faster
```

Score:

```text
Impact 3 * Confidence 4 / Effort 1 = 12.0
```

## Validation

Passed before commit:

```text
sha256sum -c artifacts/optimization/golden_checksums.txt --ignore-missing
RCH_REQUIRE_REMOTE=1 rch exec -- cargo test -p ft-device device_error_display -- --nocapture
RCH_REQUIRE_REMOTE=1 rch exec -- cargo check -p ft-device --all-targets
RCH_REQUIRE_REMOTE=1 rch exec -- cargo clippy -p ft-device --all-targets -- -D warnings
cargo fmt -p ft-device --check
git diff --check -- crates/ft-device/src/lib.rs artifacts/optimization/2026-06-05_ft_device_static_mismatch_display_frankentorch-3jnn.md .beads/issues.jsonl
ubs crates/ft-device/src/lib.rs artifacts/optimization/2026-06-05_ft_device_static_mismatch_display_frankentorch-3jnn.md .beads/issues.jsonl
```

UBS exited 0. It reported no critical issues; remaining warnings are existing
ft-device test inventory such as `expect` and assert surfaces.
