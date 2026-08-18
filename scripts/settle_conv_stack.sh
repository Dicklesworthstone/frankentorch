#!/usr/bin/env bash
# Settle every measurement this session's conv stack owes — `frankentorch-hi9r6` item 187.
#
# Items 170, 172, 173, 178 and 182 each ended "owed: a paired run". Run separately they would be
# five host states, and pairing across runs is this campaign's most repeated error (items
# 123/135/139 on pool width, 145 on pooled rows, 169 on slot profiles, 185e on staging). This runs
# them back to back on ONE snapshotted binary so at least the ELF and the window are common.
#
# It measures nothing itself and decides nothing. It enforces the four things that get forgotten
# when a quiet window finally arrives and there is a queue of runs to get through:
#
#   1. df BEFORE anything, refusing under the floor;
#   2. the ELF is SNAPSHOTTED and its sha printed, because peers rebuild the shared target
#      mid-session (item 66a: the binary changed three times under one session);
#   3. every arm-internal probe runs at BOTH pool widths, because items 165c and 170 predict
#      effects with OPPOSITE width-dependence and a single-width row cannot separate them;
#   4. loadavg and CPU idle are recorded around every run, by this script, not quoted.
#
# WHAT IT DOES NOT DO: certify anything. The h2h pass at the end prints the A/A nulls and the
# slot-0 ratio (item 169) per row; a row is quotable only if BOTH nulls pass and parity matches,
# and that judgement is the reader's.
set -u

REPEATS="${1:-3}"
OUT="${2:-/tmp}"
BIN=target/release/examples/gauntlet_lane_sweep_h2h
PY="${PYTORCH_PYTHON:-/data/tmp/torchvenv-2121/bin/python}"
FLOOR_GB=42

banner() { printf '\n=== %s\n' "$*"; }

host_state() {
  printf 'host %s | loadavg %s | ' "$(hostname)" "$(cut -d' ' -f1-3 /proc/loadavg)"
  vmstat 1 2 | tail -1 | awk '{printf "cpu_idle=%s%% iowait=%s%%\n", $15, $16}'
}

free_gb=$(df -BG --output=avail /data 2>/dev/null | tail -1 | tr -dc '0-9')
if [ -z "${free_gb}" ] || [ "${free_gb}" -lt "${FLOOR_GB}" ]; then
  echo "REFUSING: /data has ${free_gb:-?}G free, floor is ${FLOOR_GB}G" >&2
  exit 1
fi
echo "/data ${free_gb}G free (floor ${FLOOR_GB}G)"

# Item 194: refuse to START when the run queue already exceeds the machine. The harness's drift
# gate tests whether load MOVED and explicitly permits "a steady busy host"; at loadavg 88 on 64
# cores that permission is wrong -- PyTorch's arm inflated twentyfold while the drift gate said
# PASS and the A/A nulls looked calm. Recording host state after the fact does not help if the
# run should never have started, so this is a gate rather than a note.
CORES=$(nproc 2>/dev/null || echo 0)
PEAK=$(cut -d' ' -f1 /proc/loadavg)
if [ "${CORES}" -gt 0 ] && awk -v l="${PEAK}" -v c="${CORES}" 'BEGIN{exit !(l > c)}'; then
  echo "REFUSING: loadavg ${PEAK} exceeds ${CORES} cores -- the run queue is longer than the" >&2
  echo "machine, so every arm would be waiting rather than working. Wait for a quiet window." >&2
  exit 1
fi
echo "loadavg ${PEAK} against ${CORES} cores"

if [ ! -x "${BIN}" ]; then
  echo "missing ${BIN} — build it first (ONE build per project; check the slot is free):" >&2
  echo "  RCH_CARGO_WRAPPER_BYPASS=1 env -u CARGO_TARGET_DIR cargo build --release \\" >&2
  echo "      -p frankentorch-api --features fair-alloc --example gauntlet_lane_sweep_h2h" >&2
  exit 1
fi

SNAP="$(mktemp -t h2h_conv_XXXXXX)"
cp "${BIN}" "${SNAP}"
chmod +x "${SNAP}"
echo "snapshot ${SNAP}"
sha256sum "${SNAP}"
host_state

# --- arm-internal probes, both widths -------------------------------------------------------
# These carry no incumbent and no ratio, so they are honest under load and go first: if the
# toggle in item 170 turns out to be a regression, the h2h pass below need not be run at all.
for probe in gemm_tile_floor_probe conv2d_forward_width_probe; do
  if [ -x "target/release/examples/${probe}" ]; then
    banner "${probe} (arm-internal; builds its own pools, so RAYON_NUM_THREADS is not used)"
    host_state
    "target/release/examples/${probe}" 2>&1 | tee "${OUT}/${probe}.log"
  else
    echo "SKIP ${probe}: not built" >&2
  fi
done

# --- h2h, both pool widths -------------------------------------------------------------------
# Item 182's `conv2d_masked_train` is included deliberately: it is the control that separates
# "we stopped computing a discarded gradient" (item 178) from "conv2d got faster".
LANES=conv2d,conv2d_masked,conv2d_big,conv2d_big_masked,conv2d_masked_train,conv3d,conv3d_masked
for width in 8 64; do
  for i in $(seq 1 "${REPEATS}"); do
    banner "h2h RAYON_NUM_THREADS=${width} repeat ${i}/${REPEATS}"
    host_state
    RAYON_NUM_THREADS="${width}" PYTORCH_PYTHON="${PY}" FT_H2H_LANES="${LANES}" \
      "${SNAP}" 2>&1 | tee "${OUT}/h2h_w${width}_r${i}.log"
  done
done

banner "done"
host_state
echo "snapshot kept at ${SNAP} — delete it yourself once the rows are banked"
echo
echo "READING THE ROWS: quotable only if BOTH A/A nulls PASS and parity is 'match'."
echo "Item 169 prints slot0/median(slot1..3) under every row: a tight value above the null's"
echo "own band means the round's FIRST sample is cold, and FT_H2H_ROUND_WARMUP=1 (item 167)"
echo "tests whether the null follows it. A row taken under that flag is NOT comparable to any"
echo "certified standing, all of which were taken at round_warmup=0."
