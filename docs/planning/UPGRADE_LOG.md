# Dependency Upgrade Log — FrankenTorch

**Date:** 2026-07-23  |  **Language:** Rust  |  **Toolchain:** nightly (rustc 1.99.0-nightly, 2026-07-22)

Method: one dependency at a time — edit `Cargo.toml`, `cargo check --workspace --all-targets`,
then `cargo test --workspace`. Never pin backward to dodge a compile error.
Build environment: `CARGO_TARGET_DIR=/data/tmp/fleet-targets/frankentorch`, local rustc
(`RCH_CARGO_WRAPPER_BYPASS=1` — the machine's `cargo` shim otherwise force-routes to remote
build workers, which do not populate the local target dir).

## Baseline (before any change)

- `cargo check --workspace --all-targets` — `Finished \`dev\` profile ... in 36m 30s` (clean; 2 pre-existing warnings)
- `cargo test --workspace` — see table below

---

## Dependency table

| Dependency | Old | New | Breaking changes | Status |
|---|---|---|---|---|

---

## asupersync code changes

---

## Rollbacks

---

## Notes for the fleet
