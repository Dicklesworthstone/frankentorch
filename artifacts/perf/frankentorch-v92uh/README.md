# v92uh — the caching allocator as a product path, measured

`frankentorch-9pafs` found that a large share of the residual vs-PyTorch loss is
FrankenTorch re-faulting freshly `mmap`ed pages every backward — PyTorch's caching
allocator hands the same pages back, ours returns them to the system. 9pafs then
closed on a **demo** (`crates/ft-api/examples/pure_rust_caching_alloc_demo.rs`) that
proves the point by replacing `#[global_allocator]`, which is a measurement binary,
not something a library may ship.

This is the same lever shipped inside FrankenTorch instead: `ft_core::buffer_pool`,
a bounded process-global free list of `Vec<f64>`, 100% safe Rust, no `unsafe`, no C,
no global allocator taken away from the consumer.

## What was measured

`crates/ft-api/examples/gauntlet_lane_sweep_h2h` gained a `max_pool3d_nopool` lane:
the identical FrankenTorch work with `buffer_pool::set_enabled(false)`, and the same
torch op under the same second name on the incumbent side. So one binary, one
invocation, one live PyTorch arm produces both arms of the comparison, and the
incumbent rows carry a free control — PT is byte-identical code under both names, so
`PT(off)/PT(on)` must land near 1.0 or the run is not readable at all.

The per-lane medians in the summary table turned out to be the wrong estimator for a
shared host: the two lanes' medians are independent, so a load excursion lands in one
and not the other and the CI has to be wide enough to cover it. The pair is sampled
**adjacently inside each round**, so the harness now also reports a **paired**
per-round ratio (min-of-2 per round, bootstrap CI over rounds), which cancels the
common-mode excursion. That is the row to read.

## Provenance

- host: AMD Ryzen Threadripper PRO 5975WX, 32 cores / 64 threads, **1 NUMA node**,
  128 MiB L3, governor `performance`, load average 19–23 throughout (peer agents
  building; this host is never quiet)
- observed rayon threads: 64 (not requested — the default pool)
- incumbent: PyTorch **2.12.1+cpu**, self-reported by the arm in the same invocation,
  torch threads = 8
- allocator: mimalloc (`--features fair-alloc`)
- executing ELF SHA-256: `f566221758d941f12282cd1e9170d4d9c952dcf6ef76a109036e91bda0ee0ad5`
  (the pre-paired-analysis binary that produced `h2h_run1.txt` and `h2h_runs_2_5.txt`
  was `01cb66b434592479f36b6aa140519e8687820ef9e96336a1816d2f02a59a0e40`)
- raw output: `paired_runs.txt` (12 invocations), `h2h_run1.txt`, `h2h_runs_2_5.txt`

## Result — max_pool3d op work, pool ON vs OFF

Paired ratio is `off/on`, so **> 1.0 means the pool is faster**.

| invocation | paired ratio | 95% CI | rounds favouring pool | PT control | verdict |
|---|---|---|---|---|---|
| 1 | 1.198 | [1.024, 1.404] | 13/16 | 1.192 | control moved |
| 2 | 1.028 | [0.941, 1.217] | 10/16 | 1.067 | control moved |
| 3 | 1.026 | [0.861, 1.415] | 10/16 | 0.913 | control moved |
| 4 | 1.216 | [1.061, 1.377] | 13/16 | 0.783 | control moved |
| **5** | **1.108** | **[1.050, 1.383]** | **12/16** | **1.009** | **pool FASTER** |
| 6 | 1.130 | [0.979, 1.246] | 10/16 | 0.970 | undecided |
| 7 | 1.257 | [1.073, 1.453] | 13/16 | 0.816 | control moved |
| 8 | 1.138 | [1.029, 1.372] | 13/16 | 1.212 | control moved |
| **9** | **1.114** | **[1.018, 1.281]** | **13/16** | **1.020** | **pool FASTER** |
| 10 | 1.241 | [1.053, 1.404] | 12/16 | 0.908 | control moved |
| 11 | 1.283 | [1.121, 1.434] | 14/16 | 0.921 | control moved |
| 12 | 1.152 | [0.980, 1.293] | 10/16 | 0.838 | control moved |

**Quotable rows** (every gate clean, incumbent control within 5%): invocations 5 and
9, at **1.108x [1.050, 1.383]** and **1.114x [1.018, 1.281]**.

**Sign evidence, which does not depend on the CI machinery at all:** the paired ratio
came out above 1.0 in **12 of 12 independent invocations** (median 1.138, range
1.026–1.283), and **143 of 192 individual rounds** favoured the pool. Twelve
independent invocations agreeing on the sign is p ≈ 0.00024 under a fair coin. The
magnitude is worth less than the sign here — the ten runs whose incumbent control
moved are not evidence of size, only of direction.

Standing on the same runs: `max_pool3d` op work sits at roughly **4.0x PyTorch** with
the pool on. That number must NOT be differenced against the 9.39x or 7.31x recorded
on this bead earlier — those are different invocations at a different host load, and
this campaign has already been burned by exactly that comparison. The 1.11x here is
what this change is worth; the standing is where the lane is now.

## Second set, taken deliberately on a much busier host

`paired_runs_final.txt`, ELF SHA-256
`ec0c88bb6f26bbb0740687309c6bdccee22ccc01e432b3ad67b1a3fe90160405`, load average
**59–68** (roughly 3x the first set — peer agents plus this session's own workspace
test run). Twelve more invocations:

- paired ratio above 1.0 in **12 of 12** again — median **1.213**, range
  1.040–1.459, 128 of 192 rounds
- two invocations clear every gate: **1.213x [1.063, 1.626]** (control 0.992) and
  **1.459x [1.149, 1.743]** (control 1.002)
- the individual CIs are wider than the first set, exactly as a 3x busier host
  predicts; the sign is unchanged

**Across both sets: 24 of 24 independent invocations put the pool ahead.** Under a
fair coin that is p ≈ 6e-8. No single invocation is worth much on this host; the
agreement is.

## Proof the pool was actually serving, not sitting inert

Every one of the twelve final invocations printed:

```
buffer_pool: hits=31 misses=4 parked=64 buffers / 90.8 MiB
```

35 pooled requests is exactly what the wiring predicts — the `max_pool3d` lane runs
3 warmups plus 16 rounds x 2 calls = 35 backwards, and the take side is wired only
there and at the backward seed (the seed is the scalar `sum`, numel = 1, far below
`MIN_POOLED_LEN`, so it never counts). **4 misses, 31 hits**: the pool is cold for
the warmups and the first timed call, then serves every subsequent one. A flat A/B
where every take had missed would look identical in the timings and completely
different here, which is the point of printing it.

## Certified vs-PyTorch standings (this is the row that counts)

The pool table above is FrankenTorch against itself. This section is the thing
campaign law calls a row: **FrankenTorch against a live PyTorch arm sampled in the
same invocation**, A/A gate PASS, parity `match`, ELF SHA-256 self-reported from
inside the process. Raw output: `vs_pytorch_rows.txt`, 8 invocations, load average
21–22, ELF `ec0c88bb6f26bbb0740687309c6bdccee22ccc01e432b3ad67b1a3fe90160405`,
incumbent PyTorch 2.12.1+cpu (torch threads 8), 64 rayon threads observed,
governor `performance`, mimalloc.

Only PASS-gated, parity-matched rows are listed. Op work = forward + backward with
the leaf built outside the timer on both sides.

| lane | FT (ms) | PT (ms) | standing | A/A null CI |
|---|---|---|---|---|
| **max_pool3d** | 4.574 | 0.851 | **5.37x SLOWER** | [0.898, 1.349] |
| **max_pool3d** | 4.949 | 0.833 | **5.94x SLOWER** | [0.729, 1.037] |
| **max_pool3d** | 5.659 | 0.927 | **6.10x SLOWER** | [0.795, 1.384] |
| **max_pool3d** | 5.986 | 0.880 | **6.80x SLOWER** | [0.663, 1.183] |
| conv3d | 22.077 | 5.944 | 3.71x SLOWER | [0.906, 1.040] |
| conv3d | 22.355 | 6.542 | 3.42x SLOWER | [0.885, 1.055] |
| conv3d | 23.685 | 6.711 | 3.53x SLOWER | [0.792, 1.189] |
| conv3d | 23.906 | 6.753 | 3.54x SLOWER | [0.857, 1.060] |
| conv3d | 24.145 | 6.285 | 3.84x SLOWER | [0.820, 1.102] |
| conv3d | 24.998 | 6.413 | 3.90x SLOWER | [0.906, 1.128] |
| conv3d | 25.640 | 6.250 | 4.10x SLOWER | [0.874, 1.063] |
| conv3d | 26.386 | 6.679 | 3.95x SLOWER | [0.894, 1.133] |
| max_pool1d | 13.981 | 8.430 | 1.66x SLOWER | [0.824, 1.423] |
| max_pool1d | 12.982 | 6.135 | 2.12x SLOWER | [0.831, 1.290] |
| max_pool1d | 13.774 | 6.167 | 2.23x SLOWER | [0.753, 1.109] |
| max_pool1d | 14.319 | 6.190 | 2.31x SLOWER | [0.888, 1.301] |
| max_pool1d | 15.148 | 5.890 | 2.57x SLOWER | [0.656, 1.107] |
| max_pool1d | 14.034 | 5.376 | 2.61x SLOWER | [0.951, 1.534] |
| avg_pool2d | 4.537 | 1.747 | 2.60x SLOWER | [0.803, 1.187] |

**These are losses, and they are the point.** `max_pool3d` sits at **5.4–6.8x
slower than PyTorch** on op work as of ELF `ec0c88bb`. That is the campaign's
largest confirmed gap and it is still a gap.

**Do not difference this against the 9.39x or 7.31x on `frankentorch-87sz8`.**
Those are different invocations at a different host load, and this repo has already
been burned by exactly that arithmetic (`frankentorch-87sz8`'s own measurement
caution: an untouched avg_pool2d lane moved 1.833 → 1.155 ms between two sweeps,
making its ratio look 40% better for free). 5.4–6.8x is the standing *now*; the only
licensed statement about the pool's contribution is the paired within-invocation
ratio above.

## What did NOT hold up

**un3os's NUMA hypothesis is refuted, for free.** `un3os` left the many-thread write
penalty unexplained and wrote "most plausibly NUMA: the recycled block was faulted by
whichever thread touched it last, so writers on other nodes pay remote bandwidth. Not
measured." This host has **one NUMA node** (`numactl --hardware`: `available: 1 nodes
(0)`, all 64 CPUs on node 0). There is no remote node for a writer to be on, so
whatever costs the many-thread write here, it is not cross-node bandwidth. The
question un3os opened stays open; one candidate is now closed.

## Honest limits

- The pool is wired on the **take** side only in `max_pool3d`'s four dense-gradient
  backwards and the backward seed. The recycle side is generic (`Drop for
  TensorBackwardReport`, `Drop for TensorTape`), so other lanes already *park*
  buffers they cannot yet *take*. Extending the take side to `avg_pool2d`,
  `max_pool1d` and `conv3d` is the obvious next lever and is not measured here.
- Parking happens at report/tape drop, which is outside the timed region on both
  arms — as is the unpooled arm's `free`. Neither arm is charged for its teardown.
- The host was never quiet. That is why the paired estimator exists and why the sign
  across invocations is doing more work than any single CI.
