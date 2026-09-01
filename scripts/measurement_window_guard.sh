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
#
# NEVER PUT THIS IN A PIPELINE. `if guard 2>&1 | head -3; then <measure>; fi` tests the PIPELINE's
# status, which is `head`'s, which is always 0 — so the guard prints REFUSING TO MEASURE and the
# measurement runs anyway, looking guarded. That voided a full frankentorch-g0wpj decomposition and
# A/B beside another project's live benchmark (frankentorch-csdoc). The verdict is the EXIT CODE;
# consume that, never the stdout:
#
#     guard >/dev/null 2>&1 || exit 1       # correct
#     if guard >/dev/null 2>&1; then ...    # correct
#     if guard 2>&1 | head -3; then ...     # SILENTLY DEFEATED
#
# Every scripted harness in this repo gets it right by calling the guard as a bare command; every
# failure so far has been a hand-written one-liner. `feedback_exit_code_and_shell_traps` records
# the same trap in the reporting case, where it merely yields a wrong number — here it yields a
# wrong number that looks guarded, which is worse.
set -uo pipefail

MAX_LOAD="${FT_GUARD_MAX_LOAD:-35}"
MAX_IOWAIT="${FT_GUARD_MAX_IOWAIT:-10}"
MAX_LOAD_RATIO="${FT_GUARD_MAX_LOAD_RATIO:-4}"

# The stability limb lives in a library so the corpus replay exercises the shipping code path.
# shellcheck source=lib/loadavg_spread_verdict.sh
. "$(dirname "${BASH_SOURCE[0]}")/lib/loadavg_spread_verdict.sh"
while [ $# -gt 0 ]; do
    case "$1" in
        --max-load) MAX_LOAD="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done
# After the arg loop: the floor's cap is derived from MAX_LOAD, so `--max-load` has to be
# parsed before it is computed.
SPREAD_FLOOR="${FT_GUARD_SPREAD_FLOOR:-$(loadavg_spread_floor "$MAX_LOAD")}"

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
readonly COMM_EXCLUDE='^(rustc|rustfmt|clippy-driver|cc1|cc1plus|ld|lld|ld[.]lld|collect2|as|zsh|bash|sh|dash|ps|grep|awk|sed|tee|ssh|sshd|scp|rsync)$'

mapfile -t HITS < <(
    ps -eo pid=,comm=,args= 2>/dev/null | awk \
        -v pat="$ARGS_PATTERN" -v excl="$COMM_EXCLUDE" -v self="$$" -v parent="$PPID" '
        {
            pid = $1; comm = $2;
            args = $0; sub(/^[ \t]*[0-9]+[ \t]+[^ \t]+[ \t]+/, "", args);
            if (pid == self || pid == parent) next;
            if (args ~ /measurement_window_guard/) next;
            # The tick-delivery script carries the tick TEXT as its argv, and that text
            # contains the word "benchmarking" -- so it matched `bench` and the guard
            # reported the very thing that had just told it to measure. Excluded by name.
            # (No apostrophes in this block: it lives inside a single-quoted awk program.)
            if (args ~ /franken_feed[.]sh/) next;
            # A SEARCH TOOL IS NOT A MEASUREMENT, and it cannot be excluded by `comm`.
            # Observed 2026-09-01: a long-lived log-monitoring
            #   ugrep ... -E --line-buffered "V3 BUILD JOB RAN|...|ffs-mounted-kernel-bench$|rwx"
            # was reported as a live peer measurement for tens of minutes. It matched because the
            # ARGS_PATTERN word `bench` appears in the REGEX IT IS SEARCHING FOR -- exactly the
            # franken_feed case above, with a different carrier.
            #
            # `grep` is already in COMM_EXCLUDE and did not help: on this box grep IS ugrep, and
            # **ugrep overwrites its own process name with its version**, so its `comm` reads
            # `2.1.251`. Any comm-based exclusion misses it. Match the INVOKED EXECUTABLE in argv
            # instead, with or without a path.
            #
            # This cannot cause a false NEGATIVE -- a search tool is not a benchmark -- so it is
            # a safe exclusion under this file`s own rule that over-matching is the safe
            # direction. It is deliberately a short list of search/monitor tools, not a general
            # CPU threshold: a peer between phases at the instant of the check burns no CPU, and
            # missing THAT costs a banked ratio.
            if (args ~ /^([^ ]*\/)?(ugrep|ripgrep|rg|ag|ack|fd|find|tail|less|watch)( |$)/) next;
            if (comm ~ excl) next;
            # REMOTE WORK DOES NOT CONTEND HERE. On this host `cargo` on PATH is an rch
            # offload shim, so a plain `cargo test`/`cargo bench` — and anything spawned
            # through `rch exec` — executes on a remote worker and costs this box only the
            # ssh client. Observed: a peer running
            # `rch exec -- cargo test -p ffs-harness --bin ffs-mounted-kernel-bench` while
            # this host sat at loadavg 12.6 and 78% idle. Flagging those tripped the guard
            # on work that could not affect a measurement here.
            #
            # What DOES contend is a directly executed binary — `fp-bench`, an h2h binary, a
            # torchvenv python. Those are what item 244 caught at 935% CPU.
            if (comm == "cargo" || comm == "rch") next;
            if (args ~ /(^|[ \/])rch( |$)/) next;
            # ...including a `timeout N cargo ...` wrapper, whose own comm is `timeout`.
            if (args ~ /(^|[ \/])cargo( |$)/) next;
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

# IOWAIT, SAMPLED AS A DELTA. A host can sit at a modest loadavg with no peer benchmark and
# still be unmeasurable because it is disk-bound — five concurrent local builds will do it, and
# neither of the checks above can see that.
#
# It has to be a DELTA over a short window: /proc/stat's iowait field is CUMULATIVE since boot
# (it reads ~20.6 million jiffies here), so the raw number carries no information about now.
# Sampling twice and differencing is the only form of this check that means anything.
IOWAIT_BEFORE="$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8, $6}' /proc/stat)"
sleep "${FT_GUARD_IOWAIT_WINDOW:-2}"
IOWAIT_AFTER="$(awk '/^cpu /{print $2+$3+$4+$5+$6+$7+$8, $6}' /proc/stat)"
IOWAIT_PCT="$(
    echo "$IOWAIT_BEFORE $IOWAIT_AFTER" \
        | awk '{ dt = $3 - $1; dw = $4 - $2; if (dt <= 0) print "0.0"; else printf "%.1f", 100 * dw / dt }'
)"
if awk -v w="$IOWAIT_PCT" -v m="$MAX_IOWAIT" 'BEGIN { exit !(w > m) }'; then
    echo "REFUSING TO MEASURE: iowait ${IOWAIT_PCT}% exceeds ${MAX_IOWAIT}% — host is disk-bound." >&2
    STATUS=1
fi

# STABILITY, not just level. The standing orders say to prefer a window whose 1-, 5- and
# 15-minute averages are CLOSE TOGETHER over one that is merely quiet right now, and item 251
# is why: a run taken at 1-min 7.3 while the 5- and 15-min read 62 and 77 produced a first
# invocation 15% off the two that followed it. The host had just come out of a load-77 period
# and its page cache and clocks had not settled. A 1-minute average cannot see that.
#
# AND THE RATIO NEEDED AN ABSOLUTE FLOOR. A ratio cannot tell a storm from a rounding error.
# `0.73 / 2.04 / 3.36` is a 4.6x spread on a host that was at most 5% busy for the whole
# fifteen minutes, and it refused three consecutive attempts at a lane row; `1.99 / 2.93 /
# 26.02` is the same defect one size up. Both ends are now clamped up to SPREAD_FLOOR
# (nproc/8) before dividing, which subsumes the old `lo < 0.5` clamp. See
# `lib/loadavg_spread_verdict.sh` for why the floor is a property of the machine rather than
# of the windows that were refused, and `loadavg_spread_corpus_replay.sh` for what it admits.
LOAD5="$(cut -d' ' -f2 /proc/loadavg)"
LOAD15="$(cut -d' ' -f3 /proc/loadavg)"
SPREAD_READING="$(loadavg_spread_exceeds "$LOAD1" "$LOAD5" "$LOAD15" "$MAX_LOAD_RATIO" "$SPREAD_FLOOR")"
SPREAD_EXCEEDS=$?
SPREAD_RATIO="$(echo "$SPREAD_READING" | cut -d' ' -f3)"
if [ "$SPREAD_EXCEEDS" -eq 0 ]; then
    echo "REFUSING TO MEASURE: loadavg $LOAD1 / $LOAD5 / $LOAD15 spread ${SPREAD_RATIO}x exceeds ${MAX_LOAD_RATIO}x (floor ${SPREAD_FLOOR}) — the host is still settling, quiet is not stable." >&2
    STATUS=1
fi

if [ "$STATUS" -ne 0 ]; then
    echo "Guard FAILED — do source work and retry. (overrides: FT_GUARD_MAX_LOAD, FT_GUARD_MAX_IOWAIT, FT_GUARD_SPREAD_FLOOR)" >&2
    exit "$STATUS"
fi

echo "guard PASS: no peer measurement detected, loadavg $LOAD1/$LOAD5/$LOAD15 (spread ${SPREAD_RATIO}x <= ${MAX_LOAD_RATIO}x, floor ${SPREAD_FLOOR}), iowait ${IOWAIT_PCT}% <= ${MAX_IOWAIT}%"
