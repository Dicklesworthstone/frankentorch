# 87sz8 — vs-PyTorch re-measure after this session's max_pool backward work

## The honest headline: this run does NOT show the kernel wins reaching the lane, and it does NOT show a regression either. It is not comparable to the baseline.

`un3os` made four max_pool backward kernels 1.4–2.2x cheaper (and the profiled one
4.24x in an isolated A/B). Every one of those write-ups said the same thing: **no
vs-upstream ratio is claimed until the h2h runs.** This is that run, and the result
is that the question is still open.

## What was measured

`crates/ft-api/examples/gauntlet_lane_sweep_h2h.rs`,
`executing_elf_sha256=93e8de27a87f5efae34fd382c33c86be9d1e5498178ae2719dc8f16ba2505042`,
mimalloc, interleaved arms, PyTorch **2.12.1+cpu** self-reported by the arm in the
same invocation, torch threads=8.

| lane | FT (ms) | PT (ms) | standing | A/A gate | parity |
|---|---|---|---|---|---|
| `max_pool1d` | 13.002 | 8.869 | FT 1.47x slower | PASS [0.907,1.089] | match |
| `avg_pool2d` | 3.745 | 1.701 | FT 2.20x slower | PASS [0.846,1.079] | match |
| `max_pool3d` | 4.770 | 0.906 | FT 5.26x slower | PASS [0.799,1.151] | match |
| `conv3d` | 20.491 | 6.035 | FT 3.40x slower | PASS [0.878,1.075] | match |

All four lanes are internally quotable — A/A PASS, parity match.

## Why it is NOT comparable to the 4.66x baseline, and why I am not calling this a regression

**Host load was 86** during this run. The 2026-08-08 baseline that recorded
max_pool3d at 4.66x was taken at **load median 28.9**, itself already flagged BUSY.
That is a 3x difference in host contention between the two readings.

The absolute numbers say the same thing: FT's max_pool3d op work reads **4.770 ms
here against 3.483 ms in the baseline**. FT got ~1.3 ms *slower in absolute terms*
while its own backward kernel got ~1.07 ms *cheaper*. A kernel that is provably
faster cannot make the op slower; the arm moved because the machine did.

**An A/A PASS does not license a cross-run comparison.** The A/A gate certifies that
*within this invocation* the harness could resolve a difference. It says nothing
about whether two runs taken at load 28.9 and load 86 can be differenced. Treating
PASS as permission to compare across windows is exactly the error
`frankentorch-8ieqm`'s veto was built to prevent, one level up.

So: **the 5.26x here and the 4.66x baseline are not a delta.** Neither is evidence
about the other.

## The open question, stated plainly

Whether `un3os`'s kernel wins reach this lane is **unresolved**. Two possibilities,
and this run distinguishes neither:

1. They do, and load masked it.
2. They do not, because the lane's time is dominated by the forward and the tape
   rather than by the dense-gradient write the gate fixed. `pool_kernel_vs_tape_probe`
   already showed the tape is a large share of the session-level cost, so this is a
   live possibility and not a strawman.

## The measurement that WOULD settle it, and why it is different from this one

Do not re-run this harness and difference it against the baseline — that repeats the
confound. Instead run **two FT ELFs against the same PyTorch arm in the same
window**: one with `DENSE_SCATTER_PARALLEL_MIN_BYTES` active, one with
`dense_scatter_should_parallelize` flipped to always-true. The PyTorch arm and the
host state are then common-mode and cancel, exactly as the per-kernel A/Bs did. That
isolates the gate's effect *on the lane* rather than on the kernel, and it is
load-independent by construction.

That is the next step and it is not done here.

## Reproducing the PyTorch arm

The arm needs an interpreter with torch; this host's `python3` has none, which is
why the lane sweep had not been re-run. Provisioning that works:

```
uv venv /data/tmp/torchvenv-2121 --python 3.12
uv pip install --python /data/tmp/torchvenv-2121/bin/python "torch==2.12.1" \
  --index-url https://download.pytorch.org/whl/cpu
PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  /data/tmp/cargo-target/release/examples/gauntlet_lane_sweep_h2h
```

**Pin the version.** `frankentorch-wnku0` recorded that the same ELF reads 2.43x
against torch 2.12.1 and 1.29x against another version — the incumbent's version is
part of the measurement. The default `uv pip install torch` gives **2.13.0** today,
which would silently not be the baseline's incumbent. The harness self-reports the
version for this reason; check that line before quoting anything.

torch installs without numpy under this recipe and prints a `Failed to initialize
NumPy` warning. Harmless here — the arm only uses torch tensors — but it is in the
output and should not be mistaken for a fault.

## What this changes in the ledger

Nothing is claimed as improved. `max_pool3d` remains the largest confirmed
vs-upstream gap. The one durable gain is that the h2h arm is now reproducible on
this host, so the deciding measurement above can be run by anyone in a quiet window.
