# HEAD's f32 masked-conv2d fix, measured: 4.55x SLOWER → 2.71x SLOWER

**A WIN, and still a LOSS.** The worst conv2d standing in the tree moves from
**4.55x slower** than PyTorch to **2.71x slower** — a **1.636x** improvement from
one commit — and what remains is still the largest single conv2d loss on the board.

Commit `6c7aef5f` added `conv2d_backward_dinput_direct_f32` and its dispatch. It
shipped unmeasured. This is the measurement.

## The design: a twin pair in ONE invocation

`conv2d_f32_masked` runs HEAD's route. `conv2d_f32_masked_panel` is the *same lane*
with `set_conv2d_dinput_panel_legacy(true)` wrapped around the FrankenTorch arm —
the pre-fix `dpanel` + `col2im` round trip. **torch is byte-identical code under
both names**, so `PT(panel)/PT(masked)` is a free control that must land near 1.0.

That control is the check an A/A null structurally *cannot* make: an A/A null
compares two positions inside one run, so a uniformly scaled incumbent cancels out
exactly. Earlier in this session a crushed torch arm produced a "127.993x FASTER"
row whose A/A null passed at 1.011. **`PT(panel)/PT(masked) = 25.106 / 25.734 =
0.976`** — within 2.4% of 1.0, so the window did not move between the two arms.

```
FT_H2H_REPS=32 FT_H2H_LANES_EXACT=1 \
FT_H2H_LANES="conv2d_f32_masked,conv2d_f32_masked_panel,conv2d_f32,conv2d_f32_kernels" \
PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of target/release/examples/gauntlet_lane_sweep_h2h, --features fair-alloc>
```

* `executing_elf_sha256=0df4db5435ab47d398d06348922b28bf68b17bea37db289dc9422bff87b53b5d`
  — self-reported by the running process.
* Incumbent **PyTorch 2.12.1+cpu, threads=8**, self-reported in the same invocation.
* `allocator=mimalloc (--features fair-alloc)`. **Mandatory here**: the default
  glibc allocator makes our arm pay per-iteration `mmap`/`munmap` churn PyTorch's
  caching allocator never pays, and item 193b records that its omission voided
  every ratio taken without it.
* Window: idle **90.23% then 88.45%** over the 10 s before launch (mpstat, not
  loadavg — loadavg lagged this host by 100+ all session).
* `concurrent_measurements=none ACTIVE`; two name-matched peers measured at 0% CPU
  and correctly not counted.
* 32 rounds, balanced-square ABBAABBA, four live samples per arm per round.
* Shape: batch 160, in_ch 32, 32x32, k=3.

## The rows

| lane | FT ms | PT ms | standing | ratio CI | **FT A/A null** | PT null | parity |
|---|---|---|---|---|---|---|---|
| `conv2d_f32` | 26.136 | 26.147 | **1.00x — PARITY** | 1.000 [0.986,1.035] | 0.987 | 0.951 | match |
| `conv2d_f32_kernels` | 28.474 | 26.094 | 1.09x SLOWER | 0.916 [0.871,0.937] | 1.024 | **0.998 PASS** | match |
| **`conv2d_f32_masked`** (HEAD) | **69.783** | 25.734 | **2.71x SLOWER** | 0.369 [0.359,0.377] | **1.006** | 1.041 | match |
| **`conv2d_f32_masked_panel`** (pre-fix) | **114.172** | 25.106 | **4.55x SLOWER** | 0.220 [0.214,0.228] | **1.002** | 0.975 | match |

**The lever: 114.172 → 69.783 ms = 1.636x.** Both FrankenTorch A/A nulls are inside
±0.006 of 1.0, the ratio CIs are tight and non-overlapping (`[0.359,0.377]` against
`[0.214,0.228]`), and the two arms differ by exactly one boolean.

## What is certified and what is not

**Certified: the lever.** FT nulls 1.006 and 1.002, the PT twin control at 0.976,
non-overlapping ratio CIs, parity `match` on both arms. A 63% delta cannot be
manufactured by any of those.

**NOT certified: the absolute standings.** The harness prints `NULL-FAILED … do not
quote this row` against all four, and that verdict is honoured here rather than
argued past. The reason is narrow and worth stating: on the two masked lanes it is
the **incumbent's** null that misses, at 1.041 and 0.975 against a ±0.02 band — 4.1%
and 2.5% out — while *our* arm's nulls are clean. torch wobbled a few percent
between its own in-run samples. At 16 rounds those same nulls were 1.067 and 1.102;
32 rounds halved the miss, exactly as item 193c's 16→32→64 walk predicts. **A third
pass at 64 rounds is what these rows need to be banked**, and the numbers should be
read as ±4% until then.

`cpu_mhz spread=3.003x` with arms unpinned is a second reason for that caution: the
harness states the two arms are not *known* to have run at comparable clocks.

## The 4.55x claim: I doubted it, and it was right

Earlier this session I flagged the 4.55x figure in the doc comment on
`conv2d_backward_dinput_direct_f32` as unsupported — no ledger item, no null, no
window, no ELF hash, absent from all 2.5 MB of `NEGATIVE_EVIDENCE.md` and from
`artifacts/`. That criticism was about provenance and it was fair.

**The number itself replicates exactly.** The pre-fix panel route measures
**4.55x SLOWER** here, against 4.55x claimed. At 16 rounds it read 4.35x; at 32 it
lands on the claimed value to three digits. The original measurement was sound; only
its paper trail was missing. This file is that paper trail.

## Mechanism, for the record

At this lane's shape `flat` is 163,840 and `patch_width` 288, so the pre-fix
`dpanel` is 47.2M f32 = **189 MB**, written once by `sgemm` and read back once by
`conv2d_col2im_f32` — a ~378 MB DRAM round trip per call to produce a 24 MB
`dpadded`. HEAD's route blocks that into L2-resident row panels. The measured
44.4 ms saved is the right order for eliminating that round trip.

The f64 side stopped paying this at `4b157b3e`; f32 never did until `6c7aef5f`.
That is the **asymmetric-dtype fast path** shape again — a fast path gated on one
dtype stranding the other.

## What is still open on this lane

2.71x slower is the **largest single conv2d loss on the board** and the second
largest standing in the tree after the SVD square forward at n=512 (2.515x, commit
`75d3fad3`).

Two leads, neither chased here:

1. **The f64/f32 gate asymmetry survives the fix.** The f64 masked dispatch got
   `conv2d_dinput_blocked_any` (which splits channels *and* images), while the f32
   dispatch still uses `conv2d_dinput_blocked_selected` (images only). At batch 160
   the image axis fills the pool either way, so it does not bite at *this* shape —
   but it will strand smaller batches exactly as it did for f64 at batch 8.
2. **`conv2d_f32_kernels` is slower than `conv2d_f32`** — 28.474 against 26.136 ms —
   which is backwards: the kernels-only lane has no session and no tape and should
   be the cheaper of the two. Its FT null is the one that fails (1.024), and at 16
   rounds it carried a `slot0/median = 1.149` cold-first-sample flag. Possibly an
   instrument artefact, possibly real. Not chased.

## 64-round pass: the pre-fix route is now CERTIFIED, and the lever replicates three times

Re-run at `FT_H2H_REPS=64`, the step item 193c's 16→32→64 walk prescribes. Same ELF,
same lanes, idle 80.61% then 89.26% before launch.

| lane | FT ms | PT ms | standing | FT null | PT null | verdict |
|---|---|---|---|---|---|---|
| `conv2d_f32` | 26.260 | 25.704 | 1.02x SLOWER | 0.989 | 1.021 | not certified |
| `conv2d_f32_kernels` | 29.396 | 25.317 | 1.16x SLOWER | 1.059 | 0.977 | not certified |
| `conv2d_f32_masked` (HEAD) | 70.410 | 25.371 | **2.78x SLOWER** | **1.001** | 1.041 | measured, ±4% |
| **`conv2d_f32_masked_panel`** (pre-fix) | 114.179 | 25.435 | **4.49x SLOWER** | **0.994–1.018 PASS** | **0.949–1.050 PASS** | **CERTIFIED — both nulls pass** |

**`conv2d_f32_masked_panel` at 4.49x SLOWER is a banked row**: it is the only lane of
the four with no `NULL-FAILED` line at all. That is the pre-fix standing, certified.

### The lever replicates across three independent round counts

| rounds | pre-fix ms | HEAD ms | lever |
|---|---|---|---|
| 16 | 111.833 | 71.202 | 1.571x |
| 32 | 114.172 | 69.783 | 1.636x |
| 64 | 114.179 | 70.410 | **1.622x** |

Three passes, 1.57–1.64x, with the FrankenTorch null clean on both masked lanes in
every one. The win is not a window artefact.

### The HEAD lane's incumbent null has PLATEAUED — more rounds will not clear it

`conv2d_f32_masked`'s incumbent null went **1.067 → 1.041 → 1.041** across 16, 32 and
64 rounds. It stopped improving. The harness's "ROUNDS MAY CLEAR IT" advice applies
when a null is close *and* the miss is sampling noise; a value that halves once and
then sits still is **systematic**, not noise, and a 128-round pass would burn an hour
to land on 1.041 again.

What it is not: torch drifting between lanes. `PT(panel)/PT(masked) = 25.435/25.371 =
1.0025`, so the incumbent is consistent *across* lanes to 0.25%. The 4.1% offset is
*within* the masked lane, between its own two incumbent slots — and the neighbouring
lane, whose FrankenTorch arm is 114 ms rather than 70 ms, does not show it. The
plausible mechanism is that ABBAABBA places the two incumbent samples adjacent to
FrankenTorch work of different duration, so they see different cache and turbo state,
and the effect scales with how much the FT arm differs in length. That is a hypothesis
about the harness, not a measurement, and it is not chased here.

**So `2.78x SLOWER` is quoted as measured with a stated ±4%, and will not be called
certified.** The certified numbers on this lane are the pre-fix standing (4.49x) and
the lever (1.62x).
