# Validator Optimization Evidence — 2026-02-13

## Scope
- Target binary: `crates/ft-conformance/src/bin/validate_phase2c_artifacts.rs`
- Optimization lever: single-pass packet file cache (read each required packet artifact once, validate from cached payloads).
- Safety invariant: preserve strict/hardened behavior and fail-closed JSON validation semantics.

## Benchmark Corpus
- Synthetic root: `/tmp/ft_phase2c_bench_before_ep1sp0`
- Construction: 250 packet directories cloned from `artifacts/phase2c/FT-P2C-001` plus global controls.

## Commands
```bash
# Baseline and after (same command)
hyperfine --warmup 3 --runs 15 --ignore-failure \
  '/tmp/frankentorch-target/debug/validate_phase2c_artifacts /tmp/ft_phase2c_bench_before_ep1sp0 > /dev/null'

# Syscall profile
strace -c /tmp/frankentorch-target/debug/validate_phase2c_artifacts /tmp/ft_phase2c_bench_before_ep1sp0 > /dev/null
```

## Results
- Baseline mean wall-time: `68.8 ms`
- After mean wall-time: `64.7 ms`
- Wall-time delta: `~6.0%` faster

Syscall totals:
- Baseline: `12,354`
- After: `10,098`
- Delta: `~18.3%` fewer calls

## Isomorphism Evidence
- `cargo fmt --check` passed
- `cargo check --all-targets` passed
- `cargo clippy --all-targets -- -D warnings` passed
- `cargo test --workspace` passed
- `cargo test -p ft-conformance -- --nocapture` passed
- `cargo bench` passed

## Risk Notes
- Required-file existence semantics retained via `raw.is_some() || path.exists()`.
- Missing/unreadable markdown and parity-gate behavior remains unchanged.
- JSON contracts remain fail-closed when payload is malformed or absent.
