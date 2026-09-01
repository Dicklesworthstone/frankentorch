#!/usr/bin/env bash
# Replay every loadavg triple this repo has recorded through the guard's stability limb, before
# and after the absolute floor — `frankentorch-mdsmm`, torch:3's gate-fix proposal.
#
# WHY A REPLAY AND NOT AN ARGUMENT. The floor was proposed because three consecutive attempts at
# a lane row were refused at loads quieter than any window the guard has ever admitted. The
# question a gate change has to answer is not "does it admit the window I wanted" — it is what
# ELSE it admits, and whether it still refuses the window the limb was built for. That is a
# question about a corpus, so this replays one.
#
# THE CORPUS IS EVERY TRIPLE, NOT A SELECTION. `grep -rhoE 'loadavg [0-9.]+ ?/ ?[0-9.]+ ?/
# [0-9.]+' artifacts/` over the repo at 9ae19434 yields 27 guard lines; all 27 are below, with
# the file:line they came from, plus item 251's `7.3 / 52.6 / 72.9` — the window that read 15%
# off and is the whole reason the limb exists. The recorded verdict of each is in the artifact
# it came from, and the replay ASSERTS the pre-floor rule reproduces it: a replay that cannot
# reproduce history is not evidence about a change to it.
#
# Usage:  scripts/loadavg_spread_corpus_replay.sh          # human-readable split
#         scripts/loadavg_spread_corpus_replay.sh --tsv    # machine-readable rows
#
# Exits non-zero if any assertion fails.
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
. scripts/lib/loadavg_spread_verdict.sh

MAX_RATIO="${FT_GUARD_MAX_LOAD_RATIO:-4}"
FLOOR="${FT_GUARD_SPREAD_FLOOR:-$(loadavg_spread_floor "${FT_GUARD_MAX_LOAD:-35}")}"

# THE HISTORICAL RULE, kept here and only here. This is the limb as it stood at 9ae19434 —
# `lo` clamped at 0.5, `hi` unclamped. It exists so the replay can show what CHANGED rather
# than assert it. Do not call it from anything that gates a measurement.
loadavg_spread_exceeds_pre_floor() {
    awk -v a="$1" -v b="$2" -v c="$3" -v r="$4" '
        BEGIN {
            hi = a; lo = a;
            if (b > hi) hi = b; if (b < lo) lo = b;
            if (c > hi) hi = c; if (c < lo) lo = c;
            if (lo < 0.5) lo = 0.5;
            printf "%.2f\n", hi / lo;
            exit !(hi / lo > r);
        }'
}

# l1 l5 l15 | recorded verdict (REFUSE/PASS) | provenance
CORPUS=$(cat <<'ROWS'
0.73 2.04 3.36    REFUSE  artifacts/perf/frankentorch-mdsmm/conv2d_xl_standing_guardrefusal_680a1ec1_torch3.log:4
3.12 2.87 9.02    PASS    artifacts/frankentorch-mdsmm_repro.log:395
1.98 2.64 9.59    REFUSE  artifacts/frankentorch-mdsmm_repro.log:393
11.49 20.72 23.48 PASS    artifacts/perf/purplehorse-worst-loss/svd_run1.log:6
10.70 9.78 22.50  PASS    artifacts/frankentorch-37sxo_repro.log:197
11.92 10.03 22.78 PASS    artifacts/frankentorch-stale-tuning-constants-lzku6_repro.log:275
1.99 2.93 26.02   REFUSE  artifacts/frankentorch-mdsmm_repro.log:197
2.20 3.14 27.49   REFUSE  artifacts/frankentorch-valnx_repro.log:242
2.59 3.38 29.19   REFUSE  artifacts/frankentorch-valnx_repro.log:240
3.25 3.70 31.16   REFUSE  artifacts/frankentorch-valnx_repro.log:238
2.44 3.59 32.96   REFUSE  artifacts/frankentorch-valnx_repro.log:236
2.44 3.78 32.86   REFUSE  artifacts/frankentorch-mdsmm_repro.log:193
2.15 3.79 34.98   REFUSE  artifacts/frankentorch-valnx_repro.log:234
3.01 4.28 37.21   REFUSE  artifacts/frankentorch-valnx_repro.log:232
4.84 4.86 39.96   REFUSE  artifacts/frankentorch-valnx_repro.log:230
3.55 5.36 52.69   REFUSE  artifacts/frankentorch-valnx_repro.log:228
4.16 5.95 56.31   REFUSE  artifacts/frankentorch-valnx_repro.log:213
2.71 6.25 59.77   REFUSE  artifacts/frankentorch-valnx_repro.log:211
1.79 6.99 63.87   REFUSE  artifacts/frankentorch-valnx_repro.log:209
2.17 8.21 68.03   REFUSE  artifacts/frankentorch-valnx_repro.log:207
2.33 9.54 72.43   REFUSE  artifacts/frankentorch-valnx_repro.log:205
2.28 11.34 77.53  REFUSE  artifacts/frankentorch-valnx_repro.log:203
2.55 13.59 83.01  REFUSE  artifacts/frankentorch-valnx_repro.log:201
12.57 82.80 177.25 REFUSE artifacts/frankentorch-stale-tuning-constants-lzku6_repro.log:237
5.30 117.79 223.70 REFUSE artifacts/frankentorch-37sxo_repro.log:192
5.30 117.79 223.70 REFUSE artifacts/frankentorch-stale-tuning-constants-lzku6_repro.log:235
21.75 93.54 261.36 REFUSE artifacts/frankentorch-stale-tuning-constants-lzku6_repro.log:273
44.56 118.25 283.72 REFUSE artifacts/frankentorch-stale-tuning-constants-lzku6_repro.log:271
7.30 52.60 72.90  REFUSE  docs/NEGATIVE_EVIDENCE.md:38559 (item 251 — THE window this limb exists for)
ROWS
)

TSV=0
[ "${1:-}" = "--tsv" ] && TSV=1

# The loop runs in THIS shell (`done <<< ...`, not a pipe), so the counters below survive it.
# A `printf | while` would have counted in a subshell and reported zeros — the same shape of
# defect as reading a guard verdict through a pipeline.
n_total=0; n_admit=0; n_still=0; n_replay_fail=0; n_monotone_fail=0; n_251_fail=0
while read -r l1 l5 l15 recorded prov; do
    [ -z "${l1:-}" ] && continue
    n_total=$((n_total + 1))

    old_ratio="$(loadavg_spread_exceeds_pre_floor "$l1" "$l5" "$l15" "$MAX_RATIO")"
    old_refuse=$?
    new_reading="$(loadavg_spread_exceeds "$l1" "$l5" "$l15" "$MAX_RATIO" "$FLOOR")"
    new_refuse=$?
    new_ratio="$(echo "$new_reading" | cut -d' ' -f3)"

    old_v=PASS;  [ "$old_refuse" -eq 0 ] && old_v=REFUSE
    new_v=PASS;  [ "$new_refuse" -eq 0 ] && new_v=REFUSE

    flag=same
    if [ "$old_v" != "$recorded" ]; then flag=REPLAY_MISMATCH; n_replay_fail=$((n_replay_fail + 1)); fi
    if [ "$old_v" = REFUSE ] && [ "$new_v" = PASS ]; then flag=ADMITTED; n_admit=$((n_admit + 1)); fi
    if [ "$old_v" = REFUSE ] && [ "$new_v" = REFUSE ]; then n_still=$((n_still + 1)); fi
    # MONOTONICITY. Flooring can only shrink the ratio, so the new limb must refuse a strict
    # subset of what the old one did. A window that used to be admitted and is now refused would
    # mean the change tightened the gate somewhere, which is not what it claims to do.
    if [ "$old_v" = PASS ] && [ "$new_v" = REFUSE ]; then flag=MONOTONICITY_VIOLATED; n_monotone_fail=$((n_monotone_fail + 1)); fi
    case "$prov" in
        *"item 251"*) [ "$new_v" = REFUSE ] || { flag=ITEM_251_REGRESSED; n_251_fail=$((n_251_fail + 1)); } ;;
    esac

    if [ "$TSV" -eq 1 ]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$l1" "$l5" "$l15" "$old_ratio" "$new_ratio" "$old_v" "$new_v" "$flag" "$prov"
    else
        printf '  %5s %6s %7s   old %6sx %-6s   new %6sx %-6s   %-22s %s\n' \
            "$l1" "$l5" "$l15" "$old_ratio" "$old_v" "$new_ratio" "$new_v" "$flag" "$prov"
    fi
done <<< "$CORPUS"

if [ "$TSV" -eq 1 ]; then exit 0; fi

echo
echo "corpus       $n_total triples (every guard loadavg line in the repo, plus item 251's)"
echo "floor        $FLOOR   max ratio ${MAX_RATIO}x   nproc $(nproc)"
echo "ADMITTED     $n_admit  previously refused, now admitted"
echo "STILL REFUSED $n_still"
echo "replay of the pre-floor rule against recorded verdicts: $((n_replay_fail == 0 ? 1 : 0))/1 OK ($n_replay_fail mismatches)"
echo "monotonicity (new refuses a subset of old):             $n_monotone_fail violations"
echo "item 251's window still refused:                        $([ "$n_251_fail" -eq 0 ] && echo yes || echo NO)"

rc=0
[ "$n_replay_fail" -eq 0 ] || rc=1
[ "$n_monotone_fail" -eq 0 ] || rc=1
[ "$n_251_fail" -eq 0 ] || rc=1
[ "$rc" -eq 0 ] || echo "REPLAY FAILED" >&2
exit "$rc"
