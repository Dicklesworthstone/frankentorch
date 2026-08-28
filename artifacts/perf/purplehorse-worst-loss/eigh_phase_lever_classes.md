# eigh phase map with LEVER CLASSES — and a correction: its reduce is NOT memory-bound

eigh is now the worst certified single-matrix ratio in the tree (6.81-6.92x on a generic
spectrum, `04521b9c`). This prices each of its three phases on two counters plus thread scaling,
so the next lever has a stated class instead of a guess — and it corrects a generalisation I
made from SVD.

## The map

`perf record` on one binary, `FT_FIXTURE=generic`, n=512, one thread; scaling from the phase
counters at 1 vs 8 threads.

| phase | instructions | cache-misses | miss/instr | 1t -> 8t | class |
|---|---|---|---|---|---|
| `eigh_tql2_replay_blocked` | **66.18%** | 38.12% | 0.58x | **5.51x** | compute-bound, already parallel |
| `eigh_tred2_reduce` (both closures) | 26.23% | 16.39% | **0.62x** | **1.12x** | compute-bound, GATE-limited |
| `eigh_tred2_backtransform` | **3.79%** | **36.47%** | **9.6x** | **0.97x** | memory-bound |

## The correction

After finding the SVD reduction memory-bound I wrote that both reductions resist parallelism for
that reason. **That is wrong for eigh.** Its reduce is UNDER-represented in misses (0.62x), i.e.
compute-bound. It fails to scale for a different reason: `TRED2_PAR_MIN_L = 384` admits only rows
with `l >= 384` — about 128 of 512 at n=512 — so roughly 75% of the reduction is serial BY
DESIGN, not by memory pressure. And lowering the gate is measured worse (`d5cb092f`): the smaller
rows carry too little work to amortise a dispatch. So the constraint is dispatch GRANULARITY, and
the lever is a coarser parallel decomposition or an algorithmic change, not the gate and not
bandwidth.

Two ops, two reductions, two different binding constraints. The shared symptom (a reduction that
will not scale) invited one explanation and the counters refused it.

## What each class admits

**`tql2` — 66% of instructions, 38% of misses, scales 5.51x.** Compute-bound and already
parallel, so threading is spent. Its lever is algorithmic: replacing the QL iteration with
divide-and-conquer (`dstedc`). At 32.2% of the 8-thread lane on a generic spectrum that caps near
1.47x — worth having, and note this is the phase the h2h fixture reports at 0.6%, where the same
lever would cap at 1.006x and be correctly rejected.

**reduce — 26% of instructions, gate-limited.** Not bandwidth. The measured rejects (`d5cb092f`)
close the gate direction. What is untried is a decomposition that gives each task enough work to
amortise its dispatch — blocking over row RANGES rather than admitting individual small rows.

**backtransform — 3.79% of instructions, 36.47% of misses, 9.6x, scales 0.97x.** The clearest
memory-bound phase in either op. It is small in instructions and 15% of the 8-thread lane, and a
prior attempt to parallelise it measured 1.3x SLOWER across ~640 rayon dispatches — consistent
with bandwidth, not with dispatch count alone. Its lever is data movement: layout and reuse.

## NOT claimed

No vs-incumbent ratio here; these are FT-internal attributions. Absolute miss counts are inflated
by peer load (14.5 at the time, the quietest available), so the claim is RELATIVE attribution
within one process. n=1024 not run. The `dstedc` ceiling is arithmetic from the measured phase
share, not a measurement of `dstedc`.
