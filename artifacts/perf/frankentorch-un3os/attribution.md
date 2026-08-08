# un3os — where the dense-gradient scatter's time actually goes

`87sz8` modelled this term as contended first-touch page faults. `zoqws` refuted
that in situ (a serial pre-touch made the kernel 1.118x SLOWER and removed none of
the parallel pass). This partitions what is actually there.

Probe: `crates/ft-api/examples/dense_scatter_attribution.rs`. Runs 1–2 on
`executing_elf_sha256=0808484a…c97c45`, run 3 on `7dd8e4ec…9c2db6` (the same source
after `cargo fmt` reflowed three comments — whitespace only, no semantic change,
rebuilt and re-run rather than quoting a digest that no longer exists).
mimalloc, 25 reps, lanes interleaved with a rotating start, 64 rayon threads.
`max_pool3d [2,32,16,32,32] f64` → 8 MiB gradient, 131072 scattered elements
(1 in 8, which is exactly one f64 per 64-byte cache line, so **every** line of the
output is touched).

## The A/A gate vetoed the first version of this table, and that is the finding underneath the finding

The first run had **no allocator conditioning**. Its two byte-identical
`alloc_only` lanes — same code, different positions in the interleave — came out
**0.813 ms vs 0.138 ms, 83% apart**. The whole table was unusable.

The cause is not noise. `vec![0.0; n]` is `alloc_zeroed`, and mimalloc can skip
the zeroing when it knows the recycled block is already zero. So a lane that
follows a full-8-MiB writer pays a real memset, and a lane that follows a
one-element toucher pays nothing. **The probe was measuring its own lane order.**

Fix: `condition_allocator()` runs before every timed lane (outside the timed
region), allocating a same-sized block, dirtying it fully, and freeing it — so
every lane faces the same precondition, and the realistic one (in a training loop
the previous iteration's buffers were fully written before being freed). The start
lane also rotates each rep. A/A then came back **12.0%** and **0.3%** on two runs.

This is worth carrying beyond this bead: **any probe that allocates per iteration
is measuring allocator history unless it standardizes it**, and an A/A pair at two
different positions in the cycle is what exposes it.

## Result (three independent runs, A/A null 12.0% / 0.3% / 15.6%)

| lane | run 1 | run 2 | run 3 | what it isolates |
|---|---|---|---|---|
| `alloc_only` | 0.100 | 0.113 | 0.126 | lazy-allocation floor |
| `alloc_only_AA` | 0.112 | 0.113 | 0.107 | **A/A null** |
| `serial_fill` | 0.229 | 0.231 | 0.229 | 8 MiB written by ONE thread |
| `par_fill_64` | 0.648 | 0.689 | 0.834 | same bytes, 64 rayon tasks |
| `par_fill_8` | 0.434 | 0.450 | 0.438 | same bytes, 8 tasks |
| `par_fill_2` | 0.452 | 0.468 | 0.481 | same bytes, 2 tasks |
| **`kernel_scatter`** | **0.654** | **0.613** | **0.768** | **the real kernel** |
| **`scatter_serial`** | **0.318** | **0.311** | **0.348** | **same scatter body, ONE thread** |
| `scatter_store` | 0.609 | 0.587 | 0.668 | `=` instead of `+=` (prices the RMW read) |
| `scatter_usize_offsets` | 0.593 | 0.605 | 0.700 | offsets pre-cast (prices f64→usize) |

`kernel_scatter / scatter_serial` = **2.06x, 1.97x, 2.21x**. `serial_fill` is the
single most stable row in the table (0.229 / 0.231 / 0.229), which is itself a
signal: the one-thread lane barely notices the host, and the many-thread lanes are
the ones that move.

## What it says

**1. The cost is the parallelism, not the scatter.** `scatter_serial` runs the
identical body on one thread and is **1.97–2.21x FASTER** than the 64-task kernel.
That is far outside the A/A null. The same pattern shows in a pure write with no
scatter at all: `serial_fill` beats `par_fill_64` by **2.8–3.0x**.

**2. It is not rayon scheduling overhead.** The task-count sweep goes 64 → 8 → 2
and *never* reaches the single thread: 0.648 / 0.434 / 0.452 against 0.229. Two
tasks are already ~2x slower than one, and two `join`s cannot cost 0.2 ms.

**3. Neither the RMW read nor the f64→usize conversion is the story.**
`scatter_store` (0.609/0.587) and `scatter_usize_offsets` (0.593/0.605) both sit
within ~10% of `kernel_scatter` — at or inside the A/A null on run 1. Removing the
accumulate's read buys nothing measurable; removing the per-element cast buys
nothing measurable. Both candidates from the bead are **negative**.

**4. `87sz8`'s partition was right about the magnitude, wrong about the term.** The
dense materialisation really is where the time is. It is just not fault contention
— it is that writing this buffer from many threads is slower than from one.

## What is NOT established

- **Why** many-thread writes lose. The shape of the task-count sweep (even 2 tasks
  lose) points at memory placement rather than scheduling — most plausibly NUMA:
  the recycled block was faulted by whichever thread touched it last, so writers on
  other nodes pay remote bandwidth. **Not measured.** No NUMA topology was read and
  no thread pinning was tried.
- **A confound I introduced and cannot rule out:** `condition_allocator()` runs on
  the main thread, so it faults the recycled block onto *that* thread's node. That
  may bias in favour of the serial lanes. It is also what the real kernel faces
  when the allocator hands back a block the main thread last touched — so the bias,
  if present, is realistic rather than artificial. It should still be separated
  before the NUMA story is asserted.
- **Host was not quiet** (load 3.7–7.3, peer builds active), which penalises the
  many-thread lanes specifically. The direction survives — a 2x gap that reproduces
  across two runs with A/A nulls of 12.0% and 0.3% is not a load artifact — but the
  exact ratios should be re-taken quiet before being quoted anywhere load-sensitive.

## The lever this names

Serialize this kernel's dense-gradient pass, or gate its task count by size.
Predicted ~2x on `raw_bwd` for this shape, and it is **bit-identical**: the serial
lane runs the same body in the same per-plane order, planes are independent, and
no accumulation order changes.

**That prediction must be A/B'd in the kernel before it is believed.** This is a
probe, and the whole reason `un3os` exists is that `zoqws`'s probe-derived lever
inverted at the call site. Same rules: two ELFs, interleaved arms, a control
compiled into both.

Nothing here is a vs-PyTorch claim; there is no PyTorch arm in this probe.
