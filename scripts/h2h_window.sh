#!/usr/bin/env bash
# Hold the H2H measurement window EXCLUSIVELY for the duration of a command.
#
#   scripts/h2h_window.sh <command> [args...]
#
# WHY THIS EXISTS. `measurement_window_guard.sh` answers "is the host quiet RIGHT NOW", and that
# is an admission check: it cannot see a peer that starts one second later. frankentorch-8vukf
# records exactly that race — the repaired guard admitted at poll 2, and DURING the admitted
# conv2d/conv2d_masked run an `h2h-harness` peer appeared at 283% CPU. The harness detected it and
# said neither row was quotable, while load-series and drift both stayed PASS. Detection worked.
# Prevention did not exist.
#
# WHY NOT IN THE HARNESS. `harness_provenance::announce_measurement` already writes a per-pid slot
# file, and it is deliberately "advisory, never blocking". Making it blocking would not fix this:
# there are 320 `*_h2h.rs` example binaries in this repo and exactly ONE of them calls it, so the
# peer in the repro was one of the 319 that announce nothing. A Rust-side lock binds only the
# binaries you edit. The invocation layer binds all of them, and every runner already goes through
# a shell.
#
# MECHANISM: flock(1) on a single well-known file, held for the lifetime of <command>.
#   - Atomic by the kernel, so the admission-to-sampling race cannot be lost to a TOCTOU.
#   - Released by the kernel when the last fd closes, so a SIGKILLed run strands nothing. That is
#     the property a pid-file lock cannot have without liveness heuristics, and this repo already
#     carries one of those (`slot_holder_is_live`) precisely because pid files need it.
#   - The lock follows the MEASUREMENT, not this wrapper: `flock` locks an fd the command inherits.
#     Killing this script alone does NOT release it, which is correct — the measurement is still
#     running. Verified on this host, `h2h_window_witness.sh` case 4.
#
# IT ALSO RUNS THE GUARD, INSIDE THE LOCK. Acquiring first and admitting second is what removes the
# last gap between admission and sampling; see the block below. `FT_H2H_WINDOW_NO_GUARD=1` skips it
# for callers that have already admitted or do not want it.
#
# EXIT CODES. The command's own status is passed through unchanged, except:
#   75 (EX_TEMPFAIL)  the window is busy — a peer is measuring; retry later
#   76                the window was ours but the HOST is unfit — the guard refused
# A runner waiting on a peer and a runner waiting on the machine are different waits and may want
# different pacing, so they get different codes. If the wrapped command itself exits 75 or 76 it is
# reported as such; that ambiguity is accepted rather than papered over, which is why both refusals
# also print a line saying which they were.
#
# WAITING IS OPT-IN. Default is refuse-immediately, preserving this repo's standing rule that a
# measurement which waits is a measurement that hangs. `FT_H2H_WINDOW_WAIT=<seconds>` switches to a
# BOUNDED wait for callers (rotors) that would rather queue than spin.
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUARD="$REPO_ROOT/scripts/measurement_window_guard.sh"
LOCK="${FT_H2H_WINDOW_LOCK:-/data/tmp/ft-h2h-window.lock}"
HOLDER="${LOCK}.holder"
WAIT="${FT_H2H_WINDOW_WAIT:-0}"
BUSY_RC=75
UNFIT_RC=76

if [ $# -eq 0 ]; then
    echo "usage: $0 <command> [args...]" >&2
    exit 2
fi

if ! command -v flock >/dev/null 2>&1; then
    echo "h2h_window: flock(1) not found; refusing rather than running UNGUARDED." >&2
    echo "  A run that silently skips the exclusion is worse than one that does not start:" >&2
    echo "  it produces a row that looks exclusive and is not." >&2
    exit 2
fi

# The lock file is only ever an flock target and is never truncated. Create it if absent.
mkdir -p "$(dirname "$LOCK")" 2>/dev/null || true
[ -e "$LOCK" ] || : >>"$LOCK"

if [ "$WAIT" -gt 0 ] 2>/dev/null; then
    FLOCK_MODE=(-w "$WAIT")
else
    FLOCK_MODE=(-n)
fi

# `FT_H2H_WINDOW_OWNED` tells the guard that the window it is about to probe is OURS. Without it,
# the guard we run below would refuse on this script's own lock — a self-deadlock that would look
# exactly like a peer. It also makes the older pattern work:
#     guard && scripts/h2h_window.sh <elf>          (guard probes a free lock, then we take it)
#     scripts/h2h_window.sh sh -c 'guard && <elf>'  (guard sees FT_H2H_WINDOW_OWNED and skips)
export FT_H2H_WINDOW_OWNED=1
export FT_H2H_WINDOW_LOCK="$LOCK"

exec 9>>"$LOCK"
if ! flock "${FLOCK_MODE[@]}" 9; then
    echo "H2H WINDOW BUSY — refusing to start; another measurement holds the window." >&2
    if [ -s "$HOLDER" ]; then
        echo "    holder: $(cat "$HOLDER" 2>/dev/null)" >&2
    else
        echo "    holder: (no record file; the holder did not write one or is mid-startup)" >&2
    fi
    echo "    lock: $LOCK   waited: ${WAIT}s   (set FT_H2H_WINDOW_WAIT=<seconds> to queue)" >&2
    exit "$BUSY_RC"
fi

# Record who we are BEFORE admission, not after. The guard samples iowait for two seconds, and a
# peer refused during those two seconds would otherwise read an empty record and report "no record
# file" for a window that very much has a holder.
printf 'pid=%s user=%s host=%s started=%s cmd=%s\n' \
    "$$" "${USER:-?}" "$(cat /proc/sys/kernel/hostname 2>/dev/null || echo '?')" \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >"$HOLDER"
# Truncate rather than delete on exit: deleting races a concurrent reader that has already opened
# it, and an empty file reads the same as a missing one to the check above.
trap 'true >"$HOLDER" 2>/dev/null || true' EXIT

# ADMISSION HAPPENS INSIDE THE LOCK, and that ordering is the point of this script.
#
# `guard && h2h_window.sh <measure>` still leaves a gap: two callers can both pass admission and
# only then race for the window, so the loser had already decided the host was quiet. Acquiring
# FIRST and admitting SECOND collapses the two into one step — by the time this run is admitted it
# already owns the window, so nothing can start between the two. That makes
# `scripts/h2h_window.sh <measure>` a single entry point with no gap left to close by convention.
if [ -z "${FT_H2H_WINDOW_NO_GUARD:-}" ] && [ -x "$GUARD" ]; then
    if ! "$GUARD" >&2; then
        echo "h2h_window: window ACQUIRED but the HOST is not measurable; releasing without running." >&2
        exit "$UNFIT_RC"
    fi
fi

# Held, admitted, recorded. Run.
"$@"
