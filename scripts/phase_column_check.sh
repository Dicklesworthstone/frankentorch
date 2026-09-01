#!/usr/bin/env bash
# ONE COMMAND for the blocked-factorisation phase-column check — `frankentorch-ebbew`, ledger 293g.
#
# WHY THIS EXISTS. Ledger 293e-g converted every drain-style instrument in `ft-kernel-cpu` to
# thread-owned counters. Those counters have NO enable flag: the four `*_nb_sweep` harnesses simply
# call `*_stage_take_ns()` around a kernel call and print the phases. So the failure mode of that
# conversion is not a crash, it is a table of ZEROES — an instrument reading 0 for a phase that
# demonstrably ran, which is item 292's trap and is invisible to `cargo test` passing.
#
# The regression tests assert liveness per family on every gate, which is the cheap half. This is
# the other half: the phases as the SWEEPS see them, on a real host, through the real harness.
#
# WHY IT IS A SCRIPT AND NOT A CHECKLIST. The check needs a guard-admitted window, and windows on
# this host open and shut without warning: the run that produced ledger 293g had to be deferred
# because another project's build held the box for an hour. A check you can only perform by
# remembering six steps is a check that gets skipped, or worse, performed with one step missing —
# `feedback_measurement_host_identity` exists because a row was once banked without its host.
#
# WHAT IT REFUSES TO DO. It does not override the guard. If `measurement_window_guard.sh` says no,
# this exits non-zero and prints why; a window that has to be forced is not a window, and the whole
# 293 arc is about not talking a gate out of its answer. It re-guards before EVERY lane, because a
# window that was open at lane 1 may be shut by lane 4.
#
#   scripts/phase_column_check.sh [reps]     # default 12 (even, and >= the sign test's floor of 6)
#
# Output: a timestamped transcript, plus a PHASE-COLUMN VERDICT naming each family populated/ZERO.
# The ratio columns in the transcript are ordinary sweep output and carry their own 293 verdicts;
# they are NOT the subject of this check and must not be quoted from it without reading them.

set -uo pipefail

REPS="${1:-12}"
THREADS="${FT_PHASE_CHECK_THREADS:-16}"
OUT_DIR="${FT_PHASE_CHECK_OUT:-/data/tmp/ft-phase-column-check}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GUARD="$REPO_ROOT/scripts/measurement_window_guard.sh"

cd "$REPO_ROOT" || exit 2
mkdir -p "$OUT_DIR"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
LOG="$OUT_DIR/phase_columns_$STAMP.txt"

say () { echo "$@" | tee -a "$LOG"; }

say "PHASE-COLUMN CHECK  $STAMP"
say "host=$(cat /etc/hostname 2>/dev/null | tr -d '\n')  nproc=$(nproc)  rayon=$THREADS  reps=$REPS"
say "HEAD=$(git rev-parse --short HEAD 2>/dev/null)  tree_dirty=$(git status --porcelain crates/ | wc -l)"
say ""

# ---------------------------------------------------------------- guard, first and unforced
if ! "$GUARD" --max-load 35 2>&1 | tee -a "$LOG"; then
    say ""
    say "NO WINDOW — nothing measured, nothing to report. This is a result, not a failure:"
    say "  re-run when the 1-minute load is under ~35 AND the 1/5/15 spread has settled."
    say "  Do NOT pass FT_GUARD_MAX_LOAD or FT_GUARD_MAX_LOAD_RATIO to get past this."
    exit 1
fi
say ""

# ---------------------------------------------------------------- build remotely, snapshot, pin
# `feedback_snapshot_binary_before_measuring`: peers rebuild the shared target dir mid-session, so
# the ELF is copied aside and its SHA recorded. `feedback_rch_only_never_local`: builds go remote.
say "BUILD (rch, remote-required; retrying only on the 103 admission refusal)"
BUILT=0
for attempt in 1 2 3 4 5 6; do
    RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
        cargo build --release -j2 -p frankentorch-kernel-cpu \
        --example cholesky_nb_sweep --example lu_nb_sweep --example inv_nb_sweep \
        >> "$LOG" 2>&1
    rc=$?
    if [ $rc -eq 0 ]; then BUILT=1; break; fi
    if [ $rc -ne 103 ]; then say "  build failed rc=$rc — see $LOG"; exit 2; fi
    say "  attempt $attempt: rch admission refusal (103), retrying"
    sleep 20
done
[ $BUILT -eq 1 ] || { say "  rch never admitted the build"; exit 2; }

for e in cholesky_nb_sweep lu_nb_sweep inv_nb_sweep; do
    cp "target/release/examples/$e" "$OUT_DIR/$e.bin" || exit 2
    say "  ELF $e $(sha256sum "$OUT_DIR/$e.bin" | cut -c1-16)"
done
say ""

# ---------------------------------------------------------------- lanes
# Per lane: label, binary, extra arg, and the 1-based AWK fields of the PHASE columns on the
# incumbent row. Field 1 is the arm label and field 2 the median, so phases start at 3.
#   cholesky: arm median panel pred TRSM trail zero trailGF/s   -> 3 5 6   (4 is the MODEL, not a
#             measurement; `zero` is legitimately 0 in the f32 lane, so it is not asserted)
#   getrf:    arm median panel solve trail resid ...            -> 3 4 5
#   getri:    arm median forward backward resid                 -> 3 4
VERDICT_LINES=()
INCOMPLETE=0

lane () {
    local label="$1" bin="$2" arg="$3" fields="$4"
    say ""
    say "########## $label ##########"
    if ! "$GUARD" --max-load 35 >> "$LOG" 2>&1; then
        say "  GUARD REFUSED before this lane — SKIPPED (the window shut mid-run)"
        VERDICT_LINES+=("$label: SKIPPED (no window)")
        INCOMPLETE=1
        return
    fi
    say "  loadavg $(cut -d' ' -f1-3 /proc/loadavg)"
    local out="$OUT_DIR/${label// /_}_$STAMP.out"
    # shellcheck disable=SC2086
    RAYON_NUM_THREADS="$THREADS" "$bin" $REPS $arg > "$out" 2>&1
    local rc=$?
    cat "$out" >> "$LOG"
    if [ $rc -ne 0 ]; then
        say "  lane exited rc=$rc"
        VERDICT_LINES+=("$label: ERROR rc=$rc")
        INCOMPLETE=1
        return
    fi
    # Every `shipped(...)` row must carry NON-ZERO phase columns. One zero anywhere is the
    # conversion having silently switched an instrument off, which is what this check is for.
    local verdict
    verdict=$(awk -v want="$fields" '
        $1 ~ /^shipped\(/ {
            rows++
            n = split(want, f, " ")
            for (i = 1; i <= n; i++) if ($(f[i]) + 0 == 0) { zero++; where = where " col" f[i] }
        }
        END {
            if (rows == 0) print "NO INCUMBENT ROW PARSED"
            else if (zero > 0) print "ZERO PHASES in " zero " of " rows*n " cells:" where
            else print "populated (" rows " incumbent rows, all phase columns non-zero)"
        }' "$out")
    say "  PHASES: $verdict"
    case "$verdict" in
        populated*) VERDICT_LINES+=("$label: OK — $verdict") ;;
        *) VERDICT_LINES+=("$label: **$verdict**"); INCOMPLETE=1 ;;
    esac
}

lane "cholesky f64" "$OUT_DIR/cholesky_nb_sweep.bin" ""    "3 5 6"
lane "cholesky f32" "$OUT_DIR/cholesky_nb_sweep.bin" "f32" "3 5 6"
lane "getrf"        "$OUT_DIR/lu_nb_sweep.bin"       ""    "3 4 5"
lane "getri inv"    "$OUT_DIR/inv_nb_sweep.bin"      ""    "3 4"

say ""
say "================ PHASE-COLUMN VERDICT ================"
for line in "${VERDICT_LINES[@]}"; do say "  $line"; done
say ""
if [ $INCOMPLETE -eq 0 ]; then
    say "ALL FAMILIES POPULATED — the thread-owned conversion did not silence any instrument."
else
    say "INCOMPLETE OR ZEROED — read the transcript before drawing any conclusion."
fi
say ""
say "READING THE REST OF THE TRANSCRIPT. The ratio columns are ordinary interleaved sweep output"
say "and carry their own ledger-293 verdicts (paired vs marginal, exact sign test, effect against"
say "the incumbent's within-run spread, plus the A/A and knob controls). They are NOT what this"
say "check is about; a TRUSTED row here is still only an isolation result and still needs a paired"
say "lane certification before it means anything."
say ""
say "transcript: $LOG"
[ $INCOMPLETE -eq 0 ] || exit 3
