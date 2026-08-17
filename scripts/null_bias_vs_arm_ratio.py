#!/usr/bin/env python3
"""Does the FT A/A null bias depend on how close the two ARMS are in duration?

Reads every banked h2h sweep log and pairs, per lane row:
    arm ratio  FT_ms / PT_ms   (how much longer our arm is than the incumbent's)
    ft_null    our A/A null's point estimate

For rows the harness PASSed it prints no point estimate, so those are recorded as
|null-1| <= 0.02 and counted separately rather than given a fabricated value.
"""
import re
import glob
import statistics
from collections import defaultdict

ROW = re.compile(
    r"^  (\w+)\s+([\d.]+)\s+([\d.]+)\s+FT\s+[\d.]+x\s+(?:SLOWER|FASTER)\s+"
    r"PT\s+(PASS|FAIL|OFFSET|WIDE)\s+\[[^\]]*\]\s+FT\s+(PASS|FAIL|OFFSET|WIDE)\s+\["
)
NULLED = re.compile(r"NULL-FAILED: incumbent ([\d.]+), FrankenTorch ([\d.]+)")

rows = []
for path in sorted(glob.glob("*.log")):
    lines = open(path, errors="replace").read().splitlines()
    for i, line in enumerate(lines):
        m = ROW.match(line)
        if not m:
            continue
        lane, ft, pt, pt_gate, ft_gate = m.groups()
        ft, pt = float(ft), float(pt)
        if pt <= 0:
            continue
        null = None
        if i + 1 < len(lines):
            n = NULLED.search(lines[i + 1])
            if n:
                null = float(n.group(2))
        rows.append((path, lane, ft, pt, ft / pt, ft_gate, null))

print(f"parsed {len(rows)} lane rows from {len(set(r[0] for r in rows))} logs")
print()

# Bucket by how close the arms are. The hypothesis: bias appears when FT/PT is near
# 1 (arms comparable) and washes out when our arm is many times longer.
BUCKETS = [(0.0, 0.7), (0.7, 1.5), (1.5, 3.0), (3.0, 6.0), (6.0, 1e9)]
print(f"{'FT/PT arm ratio':>16}  {'n':>4}  {'FT-null PASS':>12}  {'n w/ point':>10}  "
      f"{'median null':>11}  {'frac >1':>8}")
for lo, hi in BUCKETS:
    sel = [r for r in rows if lo <= r[4] < hi]
    if not sel:
        continue
    passed = sum(1 for r in sel if r[5] == "PASS")
    pts = [r[6] for r in sel if r[6] is not None]
    med = statistics.median(pts) if pts else float("nan")
    frac = (sum(1 for p in pts if p > 1.0) / len(pts)) if pts else float("nan")
    label = f"{lo:.1f}-{hi:.1f}" if hi < 1e8 else f">{lo:.0f}"
    print(f"{label:>16}  {len(sel):>4}  {passed/len(sel):>11.0%}  {len(pts):>10}  "
          f"{med:>11.3f}  {frac:>7.0%}")

print()
print("Per-lane detail for rows whose arms are within 2x of each other:")
close = sorted((r for r in rows if 0.5 <= r[4] <= 2.0), key=lambda r: r[4])
seen = defaultdict(list)
for path, lane, ft, pt, ratio, gate, null in close:
    seen[lane].append((ratio, gate, null))
print(f"{'lane':>22}  {'n':>3}  {'FT-null PASS':>12}  {'median null (failing rows)':>26}")
for lane, vals in sorted(seen.items(), key=lambda kv: -len(kv[1])):
    pts = [v[2] for v in vals if v[2] is not None]
    passed = sum(1 for v in vals if v[1] == "PASS")
    med = f"{statistics.median(pts):.3f}" if pts else "-"
    print(f"{lane:>22}  {len(vals):>3}  {passed/len(vals):>11.0%}  {med:>26}")
