#!/usr/bin/env bash
# Regression witness for the exclusive H2H measurement window — frankentorch-8vukf.
#
#   scripts/h2h_window_witness.sh
#
# Exits non-zero if any case fails. Every case is deterministic: the outcomes asserted are
# guaranteed by the kernel's flock semantics, not by winning a scheduling race. Case 5 DOES run a
# real race, and what it asserts — exactly one winner out of N — is the property that holds however
# the race is scheduled.
#
# The bead's acceptance names two directions, and they are cases 1 and 2:
#   - an existing sampler prevents a would-be starter from beginning        (case 1)
#   - no second sampler can start after admission and contaminate a sample  (case 2, via the guard,
#     which is what every runner in this repo actually calls)
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOCK="/data/tmp/ft-h2h-window-witness.$$.lock"
export FT_H2H_WINDOW_LOCK="$LOCK"
# THE WITNESS TESTS THE LOCK, NOT ADMISSION. `h2h_window.sh` now runs the guard inside the lock,
# so without this every case that asserts rc=0 would also be asserting "this host is measurable
# right now" — and would fail on a busy host, which is exactly when someone is most likely to run
# the witness. Case 2 invokes the guard directly and case 7 forces it to refuse, so the guard path
# is still covered.
export FT_H2H_WINDOW_NO_GUARD=1
WIN=scripts/h2h_window.sh
GUARD=scripts/measurement_window_guard.sh
fails=0
pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1" >&2; fails=$((fails + 1)); }
cleanup() { rm -f "$LOCK" "${LOCK}.holder"; }
trap cleanup EXIT

echo "exclusive-window witness  lock=$LOCK"

# ---------------------------------------------------------------- 1. starter is refused
# The direction the bead states first: while a sampler holds the window, a would-be starter must
# not begin. Deterministic: the holder is confirmed to have the lock before the starter tries.
# 30s, not 5: the guard samples iowait for 2 seconds per invocation, so a holder that only
# outlives one guard call makes case 2 flaky in the direction that reads as a PASS.
"$WIN" sleep 30 >/dev/null 2>&1 &
holder=$!
for _ in $(seq 1 50); do [ -s "${LOCK}.holder" ] && break; sleep 0.1; done
if [ ! -s "${LOCK}.holder" ]; then
    fail "1. holder never took the window (setup failed, later cases are meaningless)"
else
    out="$("$WIN" true 2>&1)"; rc=$?
    if [ "$rc" -eq 75 ]; then
        case "$out" in
            *"H2H WINDOW BUSY"*pid=*) pass "1. would-be starter refused rc=75 and the holder is NAMED" ;;
            *) fail "1. refused rc=75 but did not name the holder: $out" ;;
        esac
    else
        fail "1. would-be starter got rc=$rc, expected 75 (it started INSIDE another sample)"
    fi
fi

# ---------------------------------------------------------------- 2. the guard refuses too
# This is what makes the exclusion bind on the 319 h2h binaries that know nothing about it: every
# runner does `guard && measure`, so a held window refuses them for its whole duration.
# ONE invocation, then assert on both its status and its text. Calling the guard twice to test
# two things costs 4+ seconds and lets the holder expire between them — which fails OPEN, i.e.
# looks like a pass. Capture once.
guard_out="$(env -u FT_H2H_WINDOW_OWNED "$GUARD" 2>&1)"; guard_rc=$?
if [ "$guard_rc" -eq 0 ]; then
    fail "2. guard ADMITTED while the window was held — the exclusion does not bind on plain runners"
elif printf '%s' "$guard_out" | grep -q "H2H window is held"; then
    pass "2. guard refuses while the window is held, and says why"
else
    fail "2. guard refused but not for the window: $guard_out"
fi

# ---------------------------------------------------------------- 3. self-ownership is not a deadlock
# `h2h_window.sh sh -c 'guard && elf'` must not have the guard refuse on its own wrapper's lock.
if FT_H2H_WINDOW_OWNED=1 "$GUARD" 2>&1 | grep -q "H2H window is held"; then
    fail "3. guard refused on the window it already owns — self-deadlock"
else
    pass "3. a caller inside h2h_window.sh is not refused by its own lock"
fi

# ---------------------------------------------------------------- 4. release on SIGKILL
# A pid-file lock needs liveness heuristics to survive this; flock does not. Children are collected
# BEFORE the parent dies, or they reparent to init and cannot be found by ppid.
kids=$(ps -eo pid=,ppid= | awk -v h="$holder" '$2==h {print $1}')
kill -9 "$holder" >/dev/null 2>&1
for pid in $kids; do kill -9 "$pid" >/dev/null 2>&1; done
wait >/dev/null 2>&1
sleep 1
if "$WIN" true >/dev/null 2>&1; then
    pass "4. SIGKILLing the holder releases the window (no stale lock, no reaper needed)"
else
    fail "4. window still held after the holder was SIGKILLed — a crash strands every later run"
fi

# ---------------------------------------------------------------- 5. exactly one winner in a real race
# The admission-to-sampling race itself: N starters that all passed admission, all trying at once.
# Whatever the scheduling, flock admits exactly one.
racers=8
tmp="$(mktemp -d)"
for i in $(seq 1 "$racers"); do
    ( "$WIN" sh -c 'sleep 0.4' >/dev/null 2>&1; echo "$?" >"$tmp/$i" ) &
done
wait >/dev/null 2>&1
winners=$(grep -lx 0 "$tmp"/* 2>/dev/null | wc -l)
busy=$(grep -lx 75 "$tmp"/* 2>/dev/null | wc -l)
rm -rf "$tmp"
if [ "$winners" -eq 1 ] && [ "$busy" -eq $((racers - 1)) ]; then
    pass "5. $racers concurrent starters -> exactly 1 acquired, $busy refused rc=75"
else
    fail "5. $racers concurrent starters -> $winners acquired and $busy refused (want 1 and $((racers - 1)))"
fi

# ---------------------------------------------------------------- 6. exit code passthrough
# A runner must be able to tell "the window was busy" from "my measurement failed".
"$WIN" sh -c 'exit 7' >/dev/null 2>&1
[ $? -eq 7 ] && pass "6. the wrapped command's exit code passes through" \
             || fail "6. exit code not passed through"

# ---------------------------------------------------------------- 7. admission INSIDE the lock
# The wrapper acquires first and admits second; when admission fails it must release and report
# 76, distinctly from 75, so a runner can tell "a peer is measuring" from "the host is unfit".
# Forced deterministically with an impossible load ceiling rather than by waiting for a busy host.
out7="$(env -u FT_H2H_WINDOW_NO_GUARD FT_GUARD_MAX_LOAD=0 "$WIN" true 2>&1)"; rc7=$?
if [ "$rc7" -eq 76 ]; then
    case "$out7" in
        *"HOST is not measurable"*) pass "7. guard refusal inside the lock returns 76, not 75" ;;
        *) fail "7. returned 76 without saying why: $out7" ;;
    esac
else
    fail "7. forced guard refusal returned rc=$rc7, expected 76"
fi

# And the window must be free again afterwards: a refused admission must not strand the lock.
if "$WIN" true >/dev/null 2>&1; then
    pass "8. a refused admission releases the window"
else
    fail "8. window stranded after the guard refused inside it"
fi

echo
if [ "$fails" -eq 0 ]; then
    echo "witness PASS: the window is exclusive for the DURATION of sampling, in both directions."
else
    echo "witness FAILED: $fails case(s)" >&2
fi
exit "$fails"
