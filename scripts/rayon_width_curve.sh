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
#
# USAGE
#   rayon_width_curve.sh <binary> <outfile> [widths] [lanes]
#
#     widths  space-separated, HALF the palindrome; the script mirrors it.  Default "4 8 16 32 64".
#     lanes   comma-separated, or the word `all` to drop the filter and sweep the whole board.
#
#   The DEFAULT five lanes are the curve's discovery set. The question qq8as is now blocked on is
#   the opposite one -- what the cap costs the lanes that gained nothing -- and that needs the whole
#   board at two widths:
#
#     rayon_width_curve.sh <bin> <out> "8 16" all
#
#   Note a full-board pass is ~40x the work of a five-lane pass, so mirror it once, not five times.
set -u
BIN="$1"
OUT="$2"
WIDTHS="${3:-4 8 16 32 64}"
LANES="${4:-prelu_noshortcut,avg_pool2d,max_pool3d,conv3d,max_pool1d_nopool}"

# The palindrome is built from the argument rather than hard-coded, so a two-width run gets
# 8 16 16 8 and a five-width run gets the original order. Reversal via a plain loop: `tac` on a
# here-string would drag in a subshell per pass for no benefit.
MIRROR=""
for w in ${WIDTHS}; do
  MIRROR="${w} ${MIRROR}"
done

for w in ${WIDTHS} ${MIRROR}; do
  printf 'pass width=%s load=%s\n' "$w" "$(cut -d' ' -f1-3 /proc/loadavg)"
  # PYTORCH_PYTHON is required even in ISOLATION MODE: the harness still SPAWNS the incumbent
  # process and waits for its PT_READY handshake, and only skips its timed work. Without it the
  # run aborts before a single lane is sampled.
  # `all` means NO lane filter at all, which is not the same as naming every lane: the filter is
  # what makes FT_H2H_LANES_EXACT meaningful, and an unfiltered sweep is the configuration every
  # full-board row was taken under.
  if [ "${LANES}" = "all" ]; then
    RAYON_NUM_THREADS="$w" FT_H2H_NO_INCUMBENT=1 \
      PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
      "$BIN" >> "$OUT" 2>&1
  else
    RAYON_NUM_THREADS="$w" FT_H2H_NO_INCUMBENT=1 FT_H2H_LANES_EXACT=1 FT_H2H_LANES="$LANES" \
      PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
      "$BIN" >> "$OUT" 2>&1
  fi
  printf 'PASSDONE width=%s\n' "$w" >> "$OUT"
done
echo "sweep complete -> $OUT"
