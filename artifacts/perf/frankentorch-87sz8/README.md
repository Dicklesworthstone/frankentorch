# frankentorch-87sz8 — max_pool3d forward was running single-threaded

## The defect

`POOL_FWD_PARALLEL_MIN` is `1 << 21` = 2 097 152 "input reads", and the pool
forward gate is `out.len() * kd*kh*kw >= POOL_FWD_PARALLEL_MIN`. The gauntlet's
`max_pool3d [2,32,16,32,32]` with kernel/stride 2 computes

```
out.len() * kd*kh*kw = 131072 * 8 = 1_048_576
```

— **exactly half the threshold**. So the lane that the op-work sweep measured at
9.39x slower than PyTorch was pooling 8 MiB on **one core**.

## Why the existing gate was wrong for this shape, and right for the one it was tuned on

The threshold's comment records a real measurement: on 2-D CNN feature maps a
plane is tiny and cache-resident (ResNet 512ch × 28×28 pools **784 reads per
plane**), so rayon fork/join costs more than the parallelism buys — 1.6 M reads
measured 0.81x, i.e. *slower*.

But total reads is the wrong discriminator, because it does not see how the work
divides. A 3-D pool has the opposite profile: this shape is **64 planes of 16 384
reads each**, 21x more work per rayon task than the 2-D case, and it lands below
a threshold calibrated on shapes that look nothing like it.

## The measurement that identified it

`crates/ft-api/examples/pool_kernel_vs_tape_probe.rs` — double the depth and the
shape crosses the gate:

| shape | reads | path | time |
|---|---|---|---|
| `[2,32,16,32,32]` | 1 048 576 | serial (below gate) | **2.136 ms** |
| `[2,32,32,32,32]` | 2 097 152 | parallel (at gate) | **0.946 ms** |

**Twice the data in 0.44x the time.** That is not a subtle crossover argument;
the parallel path on double the work beat the serial path outright.

## The fix

Add a per-plane clause, so the gate sees how the work divides:

```rust
const POOL_FWD_PARALLEL_PER_PLANE_MIN: usize = 1 << 12; // 4096 input reads per plane

fn pool_fwd_should_parallelize(total_reads: usize, planes: usize) -> bool {
    total_reads >= POOL_FWD_PARALLEL_MIN
        || (planes >= 2 && total_reads / planes >= POOL_FWD_PARALLEL_PER_PLANE_MIN)
}
```

Applied to the three `max_pool3d` forward gates (f64, with-indices, f32). The
second clause can only **add** parallelism, and by construction it cannot reach
the small-plane shapes the original threshold protects: at 784 reads per plane
they fail it by more than 5x. `avg_pool2d` and the other pool families are left
on the original gate — they are `frankentorch-k1h8g`'s scope, not this bead's.

Bit-exact by construction: planes are independent and each plane's arithmetic is
unchanged, so parallelising across them cannot alter a single output bit. The
first-argmax convention is per-window and equally unaffected.

## Result — FrankenTorch-side A/B, same probe, same host

| | before | after | change |
|---|---|---|---|
| `max_pool3d` raw forward | 2.136 ms | **1.083 ms** | **1.97x faster** |
| `max_pool3d` kernels (fwd+bwd) | 3.703 ms | **2.555 ms** | 1.45x faster |
| `max_pool3d` session (fwd+bwd+leaf+grad read) | 8.644 ms | 5.361 ms | 1.61x faster |

Gates: `cargo test -p ft-kernel-cpu --lib` **592 passed / 0 failed**.

## What NOT to conclude, and the control that shows why

The vs-PyTorch sweep read **9.39x before** and **7.31x after**. **Do not quote
that as the improvement.** Those are different runs, and the PyTorch arm moved
between them — on `avg_pool2d`, a lane this change does not touch at all, PyTorch
went 1.833 ms → 1.155 ms, a 1.59x swing that made that lane's *ratio* look 40%
worse (4.29x → 6.00x) with no code change whatsoever.

That untouched lane is the control, and it says plainly: **cross-run vs-PyTorch
ratios are not comparable here.** The defensible claim for this lever is the
same-binary FrankenTorch-side A/B above (1.97x on the forward), plus the
post-change standing measured against its own in-run incumbent.

## Current standing after the fix

> **SUPERSEDED 2026-08-06.** The single-run figure below was taken at `REPS=15`
> with no ELF recorded. The certified standing is now
> **7.45x SLOWER [5.85–8.53], median of 18 same-ELF runs** against torch 2.12.1
> — see `artifacts/perf/frankentorch-lane-sweep-reps16/`. The 7.31x below
> reproduces inside that range, so this section's *conclusion* stands; only its
> precision was overstated.

`max_pool3d` op work: FT 4.678 ms vs PyTorch 0.640 ms = **7.31x slower**, A/A
gate PASS `[0.765,1.181]`, gradient parity match. Still a large loss — the
forward is no longer the reason.

Remaining structure, from the post-fix probe: forward 1.083 ms, backward
1.472 ms. **The backward is now the larger half** and is the next target. It is
already `par_chunks_mut` over planes, so the lever there is not "add
parallelism"; profile it before assuming.

## A note for the sweep harness

In this run all four lanes' A/A gates PASSed, including `max_pool1d` and `conv3d`
which had FAILED in the first sweep. That makes their standings decidable for the
first time: `max_pool1d` **2.24x**, `conv3d` **3.49x**. It also means the harness
is machine-quiet-dependent rather than broken — `frankentorch-svabf` should stay
open until the arm-order randomisation lands, because a gate that passes only on
a quiet host is not yet a reliable gate.

> **Update 2026-08-06 (`lane-sweep-reps16`).** Both single-run digits above
> reproduce inside the certified ranges (`max_pool1d` 2.43x [1.14–3.18],
> `conv3d` 3.77x [3.19–4.42], 18 runs, torch 2.12.1). But the inference that a
> PASSing gate means the host was quiet is **backwards**: the A/A null is
> FT-vs-FT, so contention *widens* its CI and a wider CI brackets 1.0 more
> easily. A run reading 29.22x on `max_pool3d` passed its gate. Read CI width,
> not bracketing. Also: `max_pool1d`'s ratio is version-sensitive — it reads
> 1.29x against torch 2.13.0 purely because PyTorch regressed 1.93x on that op.
