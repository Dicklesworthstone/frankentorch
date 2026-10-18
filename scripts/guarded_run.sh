#!/usr/bin/env bash
# Wrapper that owns the measurement-window guard check and exclusive window lock — frankentorch-csdoc.
#
#   scripts/guarded_run.sh <command> [args...]
#
# Prevents inline pipeline exit-code traps (`if guard 2>&1 | head -3; then ...`, where `head` exits 0
# and silently defeats the guard) by delegating directly to `scripts/h2h_window.sh`, which acquires
# the exclusive flock and runs `measurement_window_guard.sh` as a bare command before executing
# `<command> [args...]`.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "$SCRIPT_DIR/h2h_window.sh" "$@"
