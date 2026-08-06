# 87sz8 — max_pool3d backward: the scatter loop is NOT the cost

## Negative evidence: reject the scatter-loop lever

After the forward gate fix, the backward became the larger half of `max_pool3d`
(1.511 ms vs 0.937 ms). Reading the source, the plausible target was the scatter
loop — per-element `arg_offsets[oidx] as usize` conversion plus a bounds-checked
`drow[arg] += dout[oidx]`. I labelled that inference-not-measurement at the time.
**Profiling refutes it.**

`executing_elf_sha256=62b59932df61bdd06c71acf70223e49d47daf9e8a80c814766628bc9d68ef021`,
mimalloc, 15-rep medians:

| term | ms | share of `raw_bwd` |
|---|---|---|
| `alloc_only` — the zeroed 8 MiB allocation | 0.096 | 6% |
| `alloc + touch every element` — floor for producing a dense buffer at all | **1.269** | **84%** |
| scatter work above that floor | **0.242** | **16%** |
| `raw_bwd` — the real backward | 1.511 | |

**84% of the backward is the cost of materialising a dense 8 MiB f64 gradient at
all.** The scatter loop is 16%. Optimising it perfectly — a free scatter — would
take the backward from 1.511 ms to 1.269 ms, which is still **~2x PyTorch's
entire forward+backward** (0.660 ms).

So the scatter lever is rejected on a measured ceiling, not on taste. Retry
predicate: reconsider only if the dense-materialisation term below is removed
first, at which point 0.242 ms stops being noise against it.

## Where the real wall is

Arithmetic for the whole op, post-forward-fix:

| | FrankenTorch | PyTorch |
|---|---|---|
| forward | 0.937 ms | |
| backward | 1.511 ms | |
| **total** | **2.448 ms** | **0.660 ms** |
| backward's dense-buffer floor alone | 1.269 ms | — |

Even with a free scatter *and* a free forward, FrankenTorch would sit at 1.269 ms
against PyTorch's 0.660 ms for the whole op. **The dense-gradient write is the
wall**, and it is not specific to pooling — every op whose backward returns a
dense gradient the size of its input pays it.

Rate check: 8 MiB written in 1.269 ms is ~6.3 GB/s. PyTorch's whole op moves
roughly 17 MiB (8 read + 1 write + 8 write) in 0.660 ms, ~26 GB/s. FrankenTorch's
large-buffer write path is therefore roughly **4x off** the bandwidth PyTorch
achieves on the same machine. That is the number worth attacking.

**Caveat, stated because it bounds the claim:** the 1.269 ms floor was measured
with *a* dense-write implementation (`par_chunks_mut` over 64 planes, scalar
stores), not a proven-optimal one. It is the floor for *this* write pattern. Part
of the 4x gap may be the pattern rather than an inherent limit — 64 rayon tasks
over 128 KiB chunks with scalar `f64` stores is not obviously the best way to
fill 8 MiB. Establishing the true floor (wider stores, different chunking,
non-temporal stores) is the first step of any lever here, and it should be
measured before anything is written.

## Third time the obvious target was the smaller half

Recorded because the pattern is now the most reliable finding in this bead's
lineage:

1. `ujw3g`: "optimise the pooling kernel" — refuted, the forward was 10-14% of the step.
2. `ujw3g`: "optimise leaf materialisation" — refuted, that phase was 100% caller-side copy and 0% FrankenTorch.
3. `87sz8`: "optimise the backward scatter" — refuted here, it is 16% of the backward.

Each looked like the obvious lever from reading code. In all three cases a
phase-split found the real term somewhere else. The discipline that keeps paying:
**split the phase before choosing the lever, and never on a source reading alone.**
