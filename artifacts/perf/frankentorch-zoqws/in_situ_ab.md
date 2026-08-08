# zoqws step 1 — the lever, measured in situ, is a REGRESSION

Step 0 (`bandwidth_ladder.md`) named a lever and was careful to say it had not
been shown to work at the call site:

> **NOT YET SHOWN, and this is the next step, not a conclusion:** that replacing
> `vec![0.0f64; …]` in `max_pool3d_backward_from_indices_f64` actually delivers
> this in situ.

It does not. It costs ~10%. This file is that measurement.

## What was measured

Two ELFs of `crates/ft-api/examples/pool_kernel_vs_tape_probe.rs`, differing by
**exactly** the two `din.fill(0.0)` statements landed in 5f471b62 and nothing
else (`git diff` over the whole tree during the build window showed only those
two lines; no peer edit landed between the two builds).

| arm | meaning | ELF sha256 |
|---|---|---|
| **A** | `vec![0.0; n]` → scatter (state BEFORE 5f471b62) | `405b1d4d8b6735b2852d8821a77f493952fa0fe2ec188fcd511615599edfcfbf` |
| **B** | `vec![0.0; n]` → `fill(0.0)` → scatter (state AT HEAD) | `27f3e72e3387cac954b7bee8be411223b5524777faedb8a77d1e184bb6e8defb` |

Both digests were re-read from the running process (`executing_elf_sha256`), not
from the file on disk, so the binary that produced each number is the binary
named here.

mimalloc (`--features fair-alloc`), 12 paired rounds, each round = one full run
of each arm at the probe's own 15-rep median. `max_pool3d [2,32,16,32,32] f64`,
8 MiB dense gradient. Host load 5.31 → 9.86 across the sweep.

**Arms were interleaved** (A,B,A,B…) with the **lead arm alternating each round**,
so neither arm systematically owns the cold slot.

## Result

`raw_bwd` is the edited kernel. The other three rows are compiled **identically
into both ELFs** — they are the controls, and they are what separates a real
effect from a bad measurement window.

| metric | A median [min–max] | B median [min–max] | B/A | role |
|---|---|---|---|---|
| **`raw_bwd`** | **1.353** [1.261–1.501] | **1.482** [1.419–1.636] | **1.096** | **PRIMARY — the edited kernel** |
| `ctl_touch` (8 MiB alloc + par-touch) | 1.158 [1.038–1.194] | 1.148 [1.049–1.220] | 0.992 | control |
| `ctl_avgbwd` (`avg_pool2d_backward_f64`) | 0.295 [0.290–0.312] | 0.297 [0.288–0.318] | 1.008 | control |
| `alloc_only` | 0.113 [0.090–0.120] | 0.104 [0.090–0.115] | 0.920 | control |

Paired per-round ratios (B/A on `raw_bwd`), n=12:

```
0.981 1.039 1.045 1.055 1.076 1.093 1.144 1.146 1.158 1.209 1.213 1.260
```

- paired median **1.118x SLOWER**
- bootstrap 95% CI **[1.050, 1.183]** — excludes 1.0
- sign test **11/12** rounds slower
- absolute **+0.152 ms** median
- lead-arm split: A-first rounds 1.074, B-first rounds 1.145 — both above 1.0, so
  this is not a first-slot penalty
- controls 0.992 / 1.008 — **flat**, so this is not host drift

## Why the ladder did not transfer

**+0.152 ms is almost exactly the cost of one serial 8 MiB write.** The ladder's
own warm-construction row (`vec![1.0; N]`) was 0.123 ms. So the `fill` bought
*no* reduction in the parallel pass whatsoever — it was pure added work, stacked
on top of a pass that cost the same as before.

That is the finding, and it is sharper than "the lever missed". If the parallel
pass's ~1.2 ms were contended first-touch faults, a preceding serial touch would
have absorbed them and the pass would have gotten cheaper. It did not move. So
**the contended-first-touch model does not describe this call site**, whatever it
describes about a standalone freshly-`mmap`ed buffer.

The most likely reason the two disagree is the one step 0 flagged in its own
caveat 1: the arms in a standalone ladder differ in allocator warmth, not just in
write pattern. A kernel called in a loop — which is the realistic case, and the
case the probe measures — gets a recycled block from mimalloc, so the fresh-`mmap`
premise the ladder's fast rows depended on is not what the kernel is handed.

**What is NOT established here**, and should not be asserted downstream: where
the parallel pass's ~1.2 ms actually goes. This measurement rules out "serial
touch removes it"; it does not identify the replacement explanation. A follow-up
that wants to move this term needs to attribute that 1.2 ms first, and should
treat the ladder's `serial slice::fill` row with suspicion until someone confirms
the compiler did not elide a `fill(0.0)` over `alloc_zeroed` memory — a redundant
memset over provably-zero memory is exactly the kind of thing LLVM may delete,
which would make that row measure nothing.

## Action taken

Both `din.fill(0.0)` statements reverted (`max_pool3d_backward_2x2s2_f64` and
`max_pool3d_backward_from_indices_f64`), restoring the pre-5f471b62 behaviour. A
`DO NOT RE-LAND` note carrying these numbers now sits at the call site, because
the mechanism is plausible enough that someone will otherwise re-derive it.

Bit-exactness is not in question in either direction: removing a write of zeros
over already-zero memory cannot change an output bit, which is the same argument
that justified adding it.

## Standing correction to the ledger

No vs-PyTorch number moves as a result of this. 5f471b62 never claimed a speedup
— correctly — so nothing quoted anywhere needs revising. The only thing that
changes is that `max_pool3d`'s backward is now ~10% faster than it was at HEAD
before this revert, recovering ground the unmeasured lever had silently lost.
