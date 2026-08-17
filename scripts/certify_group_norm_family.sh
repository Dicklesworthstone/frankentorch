#!/bin/sh
# Certify the group_norm family against the live PyTorch incumbent.
#
# WHY THIS SCRIPT EXISTS. The FULL 21-lane board will not certify this family: the
# incumbent's own A/A null keeps failing, so the rows come back "not quotable" no matter
# how clean the host is (NEGATIVE_EVIDENCE items 69 and 71). The fix is item 58's
# technique -- FT_H2H_LANES filters to the family, which makes a 16-round run short
# enough in wall clock that the drift gate can pass. Six runs with this recipe produced
# six drift-clean sweeps and five certifications where the full board produced none.
#
# It is kept here rather than in a scratchpad because the recipe, not the numbers, is the
# reusable part -- the numbers are in NEGATIVE_EVIDENCE item 71.
#
# USAGE:   scripts/certify_group_norm_family.sh <label> [repeats]
# EXAMPLE: scripts/certify_group_norm_family.sh g 6
#
# READ BEFORE QUOTING ANYTHING IT PRINTS:
#
#  * DO NOT certify on a busy host. Run `uptime` first. Certified rows have come back
#    2x apart on a moving machine (item 66c), and the gates do not always catch it.
#  * CHECK THE INCUMBENT'S ABSOLUTE ms ACROSS RUNS, not just the ratio. In item 71 one
#    run certified at 0.273 while the other five sat at 0.506-0.567, purely because the
#    incumbent arm read 3.610 ms instead of 6.2-6.6. The harness prints the governing
#    rule itself: a delta whose incumbent arm moved is NOT a win.
#  * THIS IS A MATCHED-THREAD-BUDGET ROW. RAYON_NUM_THREADS=8 against torch's hard-coded
#    8. It is NOT a shipped-default (64-thread) row and the two are not interchangeable.
#  * The harness must run LOCALLY; rch workers have no PyTorch.
set -u

LABEL="${1:-gn}"
REPEATS="${2:-6}"
BIN=target/release/examples/gauntlet_lane_sweep_h2h
PY="${PYTORCH_PYTHON:-/data/tmp/torchvenv-2121/bin/python}"

if [ ! -x "$BIN" ]; then
  echo "missing $BIN — build it first:" >&2
  echo "  RCH_CARGO_WRAPPER_BYPASS=1 env -u CARGO_TARGET_DIR cargo build --release \\" >&2
  echo "      -p frankentorch-api --features fair-alloc --example gauntlet_lane_sweep_h2h" >&2
  exit 1
fi

# Snapshot the ELF. The shared target directory is rebuilt by other agents mid-session;
# measuring a private copy is the only way a multi-run comparison stays on one binary
# (NEGATIVE_EVIDENCE item 66a, where the binary changed three times under one session).
SNAP="$(mktemp -t h2h_gn_XXXXXX)"
cp "$BIN" "$SNAP"
chmod +x "$SNAP"

mhz() {
  grep "^cpu MHz" /proc/cpuinfo | awk '{s+=$4; if($4<mn||mn==0)mn=$4; if($4>mx)mx=$4}
    END{printf "min=%.0f mean=%.0f max=%.0f spread=%.2fx", mn, s/NR, mx, mx/mn}'
}

echo "elf=$(sha256sum "$SNAP" | cut -d' ' -f1)"
echo "git_head=$(git rev-parse HEAD)"
echo "torch=$("$PY" -c 'import torch;print(torch.__version__)' 2>&1 | tail -1)"

i=1
while [ "$i" -le "$REPEATS" ]; do
  echo "=================== ${LABEL}${i} ==================="
  echo "pre_uptime=$(uptime)"
  echo "pre_mhz=$(mhz)"
  env RAYON_NUM_THREADS=8 FT_H2H_LANES=group_norm_f32 FT_H2H_REPS=16 \
      PYTORCH_PYTHON="$PY" "$SNAP"
  echo "post_uptime=$(uptime)"
  echo "post_mhz=$(mhz)"
  i=$((i + 1))
done

rm -f "$SNAP"
