#!/usr/bin/env bash
# The measurement guard's load-STABILITY limb, in one place so it can be replayed.
#
# WHY THIS IS A LIBRARY AND NOT INLINE IN THE GUARD. The limb below is the only part of
# `measurement_window_guard.sh` whose verdict is a pure function of three numbers, and it is
# the part that has now been wrong in both directions. Splitting it out lets
# `scripts/loadavg_spread_corpus_replay.sh` replay every loadavg triple this repo has ever
# recorded through THE CODE THE GUARD ACTUALLY RUNS. A replay against a re-typed copy of the
# rule proves nothing about the guard; that is the whole reason for the file.
#
# It is a LIBRARY, not a runnable gate: it is sourced, checks no peers, no iowait and no load
# level, and prints nothing that could be read as permission to measure. `guard --explain 1 1 1`
# was the alternative and it was rejected — it would have made
# `guard --explain 1 1 1 && measure` an exit-0 path that measures unguarded, which is the same
# family of silent defeat as the pipeline trap documented in the guard's own header.

# loadavg_spread_exceeds L1 L5 L15 MAX_RATIO FLOOR
#
# Exit 0  => REFUSE (the spread is evidence the host is still settling).
# Exit 1  => admit.
# Prints  "<hi> <lo> <ratio>" on stdout so the caller can report what it decided on.
#
# THE FLOOR, and why the limb needed one. The rule is a RATIO, and a ratio has no idea how big
# a machine is. `0.73 / 2.04 / 3.36` on a 64-core host is a 4.6x spread across a window in which
# the machine was at most 5% busy — there is no storm for it to be settling from, and yet three
# consecutive attempts (torch:3, frankentorch-mdsmm) were refused on it, at loads quieter than
# any window this guard has ever admitted. `1.99 / 2.93 / 26.02` is the same defect one size up.
#
# The fix: the limb DOES NOT APPLY when every one of the three averages is below FLOOR. A spread
# among loads that are all negligible is not evidence of anything. Above that the ratio applies
# exactly as before, unchanged, including the old `lo < 0.5` clamp.
#
# THE FIRST SHIPPED FORM WAS DIFFERENT AND WRONG. It clamped both ends up to FLOOR and divided,
# which turned 1.99/2.93/26.02 into 3.25x and admitted it — a host that averaged ~58% of itself in
# the ten minutes before. `frankentorch-wzhem` names that triple as one that must stay refused, and
# the guard-wide asymmetry says the same: over-refusing costs a tick, under-refusing costs a banked
# ratio that is a contention artefact. Gating the limb fixes the absolutely-quiet case the bead
# asked for WITHOUT admitting the settling case it asked to keep.
#
# The floor is a property of the MACHINE (nproc/8, i.e. 12.5% of it: 8 on this 64-core host, 1
# on an 8-core box), not a number chosen to admit the windows that were refused. Choosing it
# from the corpus of refusals would be fitting the gate to the complaints.
#
# WHAT IT MUST NOT UNDO: item 251's `7.3 / 52.6 / 72.9` — the window that read 15% off and is
# the reason this limb exists. Floored at 8 that is 72.9/8 = 9.1x, still refused, and the
# corpus replay asserts it.
#
# The change is MONOTONE: flooring can only shrink the computed ratio, so the new limb refuses
# a strict subset of what the old one refused. It can admit a bad window; it cannot start
# refusing a window that used to be fine. The replay asserts that too.
loadavg_spread_exceeds() {
    awk -v a="$1" -v b="$2" -v c="$3" -v r="$4" -v f="$5" '
        BEGIN {
            hi = a; lo = a;
            if (b > hi) hi = b; if (b < lo) lo = b;
            if (c > hi) hi = c; if (c < lo) lo = c;
            if (lo < 0.5) lo = 0.5;
            # THE FLOOR GATES WHETHER THE LIMB APPLIES AT ALL. It does not rescale the ends.
            # Clamping BOTH ends up to f was the first shipped form and it was too permissive:
            # 1.99/2.93/26.02 became 26.02/8 = 3.25x and was admitted, on a host that averaged
            # ~58% of itself five to fifteen minutes earlier. frankentorch-wzhem names that exact
            # triple as one that MUST STAY REFUSED, and the guard-wide asymmetry agrees — a false
            # refusal costs a deferred tick, a false admission costs a banked contention artefact.
            if (hi < f) { printf "%.2f %.2f 1.00\n", hi, lo; exit 1; }
            printf "%.2f %.2f %.2f\n", hi, lo, hi / lo;
            exit !(hi / lo > r);
        }'
}

# loadavg_spread_floor MAX_LOAD
#
# nproc/8, at least 1, and never more than a quarter of the level limb's own ceiling. The cap
# matters on a very wide host: an uncapped nproc/8 on a 1024-core box would be 128, well past
# MAX_LOAD, and a floor that large would leave the spread limb unable to fire on anything the
# level limb had not already caught. Overridable with FT_GUARD_SPREAD_FLOOR.
loadavg_spread_floor() {
    local max_load="$1"
    awk -v n="$(nproc 2>/dev/null || echo 8)" -v m="$max_load" '
        BEGIN {
            f = n / 8;
            cap = m / 4;
            if (f > cap) f = cap;
            if (f < 1) f = 1;
            printf "%.2f\n", f;
        }'
}
