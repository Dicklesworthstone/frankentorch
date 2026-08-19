#!/usr/bin/env bash
# frankentorch-rayon-pool-width-qq8as: the width CURVE, which the bead names as the only shape still
# unmeasured. Arm-internal (FT_H2H_NO_INCUMBENT), so there is no incumbent, no ratio and no drift
# gate — this asks what OUR arm costs at each width, not whether we beat anyone.
#
# PALINDROME ORDER, per item 51: a palindrome cancels a trend that reverses but not one that only
# moves one way, and this host has been falling monotonically all session. Passes run
# 4,8,16,32,64,64,32,16,8,4 so each width appears once in each direction and a monotone ramp lands
# symmetrically on all of them.
#
# One ELF, one window, RAYON_NUM_THREADS the only variable.
set -u
BIN="$1"
OUT="$2"
LANES=prelu_noshortcut,avg_pool2d,max_pool3d,conv3d,max_pool1d_nopool

for w in 4 8 16 32 64 64 32 16 8 4; do
  printf 'pass width=%s load=%s\n' "$w" "$(cut -d' ' -f1-3 /proc/loadavg)"
  # PYTORCH_PYTHON is required even in ISOLATION MODE: the harness still SPAWNS the incumbent
  # process and waits for its PT_READY handshake, and only skips its timed work. Without it the
  # run aborts before a single lane is sampled.
  RAYON_NUM_THREADS="$w" FT_H2H_NO_INCUMBENT=1 FT_H2H_LANES_EXACT=1 FT_H2H_LANES="$LANES" \
    PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
    "$BIN" >> "$OUT" 2>&1
  printf 'PASSDONE width=%s\n' "$w" >> "$OUT"
done
echo "sweep complete -> $OUT"
