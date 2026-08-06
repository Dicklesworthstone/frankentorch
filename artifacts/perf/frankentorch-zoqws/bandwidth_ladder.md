# zoqws step 0 — what dense-write bandwidth is actually achievable here

## Why this ran before any kernel was touched

`87sz8` measured 84% of `max_pool3d`'s backward as dense-gradient materialisation
at ~6.3 GB/s, versus PyTorch's ~26 GB/s for its whole op. That 6.3 was explicitly
recorded as the floor for **one** write implementation, not a proven optimum.
Bounding the phase before choosing the lever means establishing the achievable
number first — otherwise a kernel change is a guess with a benchmark attached.

## Result

`crates/ft-api/examples/dense_write_bandwidth_ladder.rs`,
`executing_elf_sha256=be00e0a403d876e99062ac973ebbb369813b5df4d1bf48bed8bfd9a363f65c4e`,
mimalloc, 8 MiB f64 (the exact buffer the pooling backward returns), 21-rep
medians, 64 rayon threads. Host was **not** quiet — load average 22 — which is
noted rather than hidden; see the caveat on that below.

| pattern | ms | GiB/s |
|---|---|---|
| `alloc_zeroed`, never touched (not a write; floor) | 0.094 | 83.32 |
| `par_chunks_mut` 128 KiB, scalar stores **[CURRENT KERNEL PATTERN]** | 1.222 | **6.39** |
| `par_chunks_mut` 128 KiB, `slice::fill` | 1.301 | 6.01 |
| `par_chunks_mut` 16 KiB, `slice::fill` | 1.065 | 7.34 |
| `par_chunks_mut` 64 KiB, `slice::fill` | 1.240 | 6.30 |
| `par_chunks_mut` 512 KiB, `slice::fill` | 1.064 | 7.35 |
| `par_chunks_mut` 2 MiB, `slice::fill` | 0.993 | 7.87 |
| `par_chunks_mut` 128 KiB, `wide` f64x4 stores | 1.199 | 6.52 |
| **serial `slice::fill`, ONE thread** | **0.213** | **36.70** |
| `vec![1.0; N]` (non-zero init at construction) | 0.123 | 63.34 |
| `par_chunks_mut` 128 KiB, write 1-in-8 (pool scatter shape) | 1.244 | 6.28 |

## What this says, and it is not what the phrasing of the bead assumed

**Parallelism is not the fix — it is the problem.** Every parallel variant sits at
6–8 GiB/s regardless of chunk size (16 KiB through 2 MiB) or store width (scalar,
`fill`, explicit `f64x4`). One thread doing the dumbest possible `slice::fill`
reaches **36.70 GiB/s, 5.7x faster than all of them**. Widening stores bought
nothing (6.52 vs 6.39); resizing tasks bought at most 1.2x.

The mechanism the numbers point at: `vec![0.0f64; N]` is `alloc_zeroed`. At 8 MiB
that is served by fresh `mmap` zero pages, so **every call faults in the whole
buffer**, and 64 threads faulting the same fresh mapping simultaneously contend
on the kernel's page-table/mmap locks. Serial fill faults the same pages with no
contention. `vec![1.0; N]` is faster still (63 GiB/s) because a non-zero init
takes the allocate-then-fill path, which can reuse a **recycled, already-faulted**
block — no faults at all, and 8 MiB is L3-resident on this host.

So the 1-in-8 scatter row is the tell: at 6.28 GiB/s it costs the same as writing
*every* element in parallel. The kernel is not paying for stores. It is paying for
first-touch page faults on a freshly zeroed mapping.

## Two caveats that bound this before anyone acts on it

1. **The arms differ in allocator warmth, not just in write pattern.** The fast
   rows (`serial fill`, `vec![1.0; N]`) benefit from block recycling that the
   `alloc_zeroed` rows cannot get, because a zeroing allocator cannot hand back a
   dirty block without zeroing it. The 9.91x "headroom" the probe prints is
   therefore an upper bound on an idealised change, **not** a speedup anyone has
   demonstrated in a kernel. Treat 5.7x (serial-fill vs current) as the honest
   ceiling and expect less.
2. **Host load average was 22 during this run.** That inflates the parallel rows
   specifically, since they compete for cores while the serial row does not.
   The direction of the finding survives — a 5.7x gap is far outside what
   contention explains, and the chunk-size sweep is internally consistent — but
   the exact ratios should be re-measured quiet before they are quoted anywhere
   load-sensitive.

## The lever this names, and what has NOT been shown

Stop asking the allocator for zeroed memory on this path. Allocate uninitialised
and zero it deliberately, so the buffer can come from a recycled warm block
instead of a fresh mapping.

This repo already has the pattern and the helper: `build_uninit` in
`ft-kernel-cpu` (the `expand`/`broadcast_to` fix used exactly this — `vec![v; numel]`
serial first-touch replaced by an uninit buffer whose fill does the first touch,
2.2–5.03x, bit-exact). `frankentorch-1zrvy` de-duplicated the crate's uninit sites
onto that helper, so there is an established, reviewed home for it.

**NOT YET SHOWN, and this is the next step, not a conclusion:** that replacing
`vec![0.0f64; …]` in `max_pool3d_backward_from_indices_f64` actually delivers this
in situ. The kernel needs zeros in the 7-of-8 elements it does not scatter into,
so it must still zero — the claim is only that zeroing a recycled block is cheaper
than being handed a fresh zeroed mapping. That has to be measured in the kernel,
same-binary A/B, bit-exact, before it is believed.

## Relationship to 3i7c0

`3i7c0` rejected in-session buffer pooling because swapping the *allocator*
(mimalloc) did not move realistic reuse lanes. This finding is compatible and
sharper: the cost is not which allocator you use, it is **asking for zeroed pages
at all**, which defeats recycling in any allocator. That is a code-side change,
not an allocator choice, so it does not reopen 3i7c0's rejected option D.
