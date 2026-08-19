#!/usr/bin/env bash
# Refuse to measure while another agent is measuring.
#
# WHY THIS EXISTS. The orchestrator made measurements one-at-a-time fleet-wide via agent-mail's
# `acquire_build_slot`. That tool returns "Build slots are disabled. Enable WORKTREES_ENABLED"
# on every call, across four ticks, so the rule cannot bind and the fleet collides.
#
# THE OBSERVED DEFECT THIS GATES. NEGATIVE_EVIDENCE item 241: I ran a vs-PyTorch SVD lane in a
# window I had "checked", and a peer's `h2h_det.bin` plus the full torch board arm were running
# throughout. The rows were unusable — FT block spread 5.3x, and phase shares 30-37%/53-61%
# against the 70-72%/11-12% three earlier invocations agreed on. The check had actually LISTED
# those processes and then printed `(clear)` on a line that ran unconditionally, so I read a
# reassuring label over the top of the evidence. This one EXITS NON-ZERO instead: a guard whose
# output does not change with what it found is not a guard.
#
# AND IT MUST NOT CRY WOLF. The first version matched `rustc` compiling any crate that merely
# DEPENDS on criterion, and matched shell wrappers whose command line quoted these words. A
# guard with false positives gets ignored, which returns you to having no guard. So the match is
# on the process's own executable (`comm`), and compilers and shells are excluded by name.
#
# DELETION CONDITION: delete this when `acquire_build_slot` works, and use the slot instead.
#
# Usage:  scripts/measurement_window_guard.sh && <your timed run>
#         scripts/measurement_window_guard.sh --max-load 30 && ...
set -uo pipefail

MAX_LOAD="${FT_GUARD_MAX_LOAD:-35}"
while [ $# -gt 0 ]; do
    case "$1" in
        --max-load) MAX_LOAD="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

# A measurement is: a torch interpreter, a bench/h2h binary, or `cargo bench`. It is NOT a
# compiler, and not a shell that happens to mention one of these words on its command line.
# `bench` UNANCHORED, not `_bench`: the first version missed a peer's `fp-bench` at 935% CPU
# because the name is hyphenated, and missed `python3 benches/vs_pandas_harness.py` because the
# word sits in a path. Only the load ceiling saved that tick. Over-matching is the safe
# direction — a false positive costs a deferred tick, a false negative costs a banked ratio
# that is a contention artefact.
#
# But NOT a bare `torch`: it matches "frankentorch", so every process in this repo — including
# the orchestrator's own `ntm internal-monitor frankentorch` — would trip the guard, and a guard
# that always fires is a guard that gets ignored. `torchvenv` is the incumbent arm's actual
# signature.
readonly ARGS_PATTERN='torchvenv|bench|h2h|criterion|rayon_width|pytorch_gauntlet'
readonly COMM_EXCLUDE='^(rustc|cc1|cc1plus|ld|lld|ld[.]lld|collect2|as|zsh|bash|sh|dash|ps|grep|awk|sed|tee)$'

mapfile -t HITS < <(
    ps -eo pid=,comm=,args= 2>/dev/null | awk \
        -v pat="$ARGS_PATTERN" -v excl="$COMM_EXCLUDE" -v self="$$" -v parent="$PPID" '
        {
            pid = $1; comm = $2;
            args = $0; sub(/^[ \t]*[0-9]+[ \t]+[^ \t]+[ \t]+/, "", args);
            if (pid == self || pid == parent) next;
            if (args ~ /measurement_window_guard/) next;
            if (comm ~ excl) next;
            # `cargo` is a compiler driver except when it is running a bench.
            if (comm == "cargo" && args !~ /bench/) next;
            if (args ~ pat) printf "%s  %s\n", pid, substr(args, 1, 100);
        }'
)

STATUS=0

if [ "${#HITS[@]}" -gt 0 ]; then
    echo "REFUSING TO MEASURE: ${#HITS[@]} peer measurement process(es) are live." >&2
    for h in "${HITS[@]}"; do
        echo "    $h" >&2
    done
    STATUS=1
fi

LOAD1="$(cut -d' ' -f1 /proc/loadavg)"
if awk -v l="$LOAD1" -v m="$MAX_LOAD" 'BEGIN { exit !(l > m) }'; then
    echo "REFUSING TO MEASURE: 1-minute loadavg $LOAD1 exceeds $MAX_LOAD." >&2
    STATUS=1
fi

if [ "$STATUS" -ne 0 ]; then
    echo "Guard FAILED — do source work and retry. (override: FT_GUARD_MAX_LOAD)" >&2
    exit "$STATUS"
fi

echo "guard PASS: no peer measurement detected, loadavg $LOAD1 <= $MAX_LOAD"
