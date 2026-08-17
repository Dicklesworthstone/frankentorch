#!/usr/bin/env python3
"""Where in the balanced square does the summed route's FT-null bias live?

The A/A null is a ratio of two slot groups, so a one-sided null means one group is
systematically slower. This prints each slot position's median so the bias can be
located rather than inferred.
"""
import re
import sys
import statistics

LINE = re.compile(
    r"SLOTS lane=(\S+) ft=\[([\d. ]+)\] pt=\[([\d. ]+)\]"
)

lanes = {}
for path in sys.argv[1:]:
    for line in open(path, errors="replace"):
        m = LINE.search(line)
        if not m:
            continue
        lane, ft, pt = m.groups()
        ft = [float(v) for v in ft.split()]
        pt = [float(v) for v in pt.split()]
        lanes.setdefault(lane, []).append((ft, pt))

for lane, rows in lanes.items():
    n = len(rows)
    ft_cols = list(zip(*[r[0] for r in rows]))
    pt_cols = list(zip(*[r[1] for r in rows]))
    fm = [statistics.median(c) for c in ft_cols]
    pm = [statistics.median(c) for c in pt_cols]
    print(f"{lane}  (n={n} rounds)")
    print(f"    FT slot medians   {fm[0]:8.3f} {fm[1]:8.3f} {fm[2]:8.3f} {fm[3]:8.3f}   ms")
    print(f"      vs slot0        {'   1.000':>8} {fm[1]/fm[0]:8.3f} {fm[2]/fm[0]:8.3f} {fm[3]/fm[0]:8.3f}")
    print(f"    PT slot medians   {pm[0]:8.3f} {pm[1]:8.3f} {pm[2]:8.3f} {pm[3]:8.3f}   ms")
    print(f"      vs slot0        {'   1.000':>8} {pm[1]/pm[0]:8.3f} {pm[2]/pm[0]:8.3f} {pm[3]/pm[0]:8.3f}")
    # The two groupings a null could plausibly use.
    first_last = statistics.median([r[0][0] for r in rows]) / statistics.median([r[0][3] for r in rows])
    halves = (statistics.median([r[0][0] for r in rows] + [r[0][1] for r in rows])
              / statistics.median([r[0][2] for r in rows] + [r[0][3] for r in rows]))
    print(f"    FT slot0/slot3 {first_last:.3f}   FT firsthalf/secondhalf {halves:.3f}")
    print()
