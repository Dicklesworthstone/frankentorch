# un3os step 2 — the size gate, measured in the kernel: 4.24x

`attribution.md` found that the dense-gradient scatter's cost is the **parallelism
itself** below the per-core L3 slice, and predicted ~2x from serialising it. Per
the `zoqws` lesson, a probe-derived prediction is not a result. This is the
in-kernel A/B.

## Arms

Two ELFs of `crates/ft-api/examples/pool_kernel_vs_tape_probe.rs` differing by
exactly the gate condition in `max_pool3d_backward_from_indices_f64`:

| arm | driver selection | ELF sha256 |
|---|---|---|
| **A** | `if true \|\| …` — always parallel (state before the lever) | `090f0cbc11a9c36e…` |
| **B** | `if dense_scatter_should_parallelize(din.len())` — serial at 8 MiB | `3a748f13343e1d92…` |

12 paired rounds, arms interleaved with the lead arm alternating.

## Result

| metric | A (always parallel) | B (gated → serial) | B/A | role |
|---|---|---|---|---|
| **`raw_bwd`** | **1.389** [1.207–1.556] | **0.321** [0.302–0.361] | **0.236** | **PRIMARY** |
| ctl: 8 MiB alloc+par-touch | 1.293 | 1.268 | 0.980 | control |
| ctl: `avg_pool2d_backward_f64` | 0.331 | 0.345 | 1.044 | control |
| ctl: `alloc_only` | 0.106 | 0.102 | 0.958 | control |

Paired per-round B/A: `0.194 0.201 0.213 0.214 0.221 0.226 0.246 0.250 0.253
0.258 0.264 0.270`

- paired median **0.236 → 4.24x FASTER**
- bootstrap 95% CI on the ratio **[0.214, 0.255]** — nowhere near 1.0
- **12/12** rounds faster
- lead-arm split **0.236 / 0.235** — no slot effect
- all three controls flat (0.958–1.044)

## The prediction and the result disagree, and that is reported rather than smoothed

`attribution.md` predicted **~2x**; in situ it is **4.24x**. Both are wins, but
they are not the same number and it would be dishonest to quote only whichever
suits.

The difference is allocator conditioning. The attribution probe deliberately
standardises allocator state before every timed lane (that is what fixed its A/A
veto), which makes both of its arms cheaper. `pool_kernel_vs_tape_probe` loops the
call the way a training step does, with no conditioning — and the parallel arm
suffers more from that than the serial arm does. So:

- **4.24x** is what the change does at this call site under realistic repetition.
- **~2x** is what it does when allocator state is held constant.

The realistic figure is the larger one, which is a reason to be *more* careful
about it, not less. Anyone re-measuring under a conditioned harness should expect
~2x and should not read that as a regression.

## Bit-exactness

Not an argument, a test:
`max_pool3d_backward_from_indices_serial_and_parallel_drivers_are_bit_identical`
runs the parallel driver by hand over the same inputs and compares **bit patterns**,
on a tie-heavy input (so a changed argmax shows) whose `dout[0]` is `-0.0` (so a
dropped accumulate shows, since `-0.0 + 0.0 == +0.0`).

**Mutation-verified rather than assumed.** Replacing the accumulate with a plain
store made it fail exactly where designed:

```
left:  …, 9223372036854775808, …   (0x8000000000000000 = -0.0)
right: …, 0, …                     (+0.0)
```

so the test observes the real sign bit and is not tautological. Mutation reverted;
596 passed / 0 failed after.

`dense_scatter_gate_sits_at_sixteen_mib` locks where the gate is, including that
the gauntlet's 8 MiB lane stays serial and that 32 MiB parallelises.

## Scope and what is NOT claimed

- **One call site.** Only `max_pool3d_backward_from_indices_f64` is gated. The same
  `vec![0.0; n]` + `par_chunks_mut` shape exists elsewhere in `ft-kernel-cpu`
  (including `max_pool3d_backward_2x2s2_f64` immediately above it and the scalar
  variant below). They are NOT touched here — widening an effect measured at one
  site is how `zoqws` went wrong, in the other direction.
- **This host.** The 32 MiB crossover is the per-CCD L3 of a Threadripper PRO
  5975WX. The gate at 16 MiB leaves margin, but a machine with much smaller L3
  slices would want it lower. It is a compile-time constant, not a runtime probe.
- **No vs-PyTorch ratio is claimed.** This A/B has no PyTorch arm. It moves
  `raw_bwd` from 1.389 to 0.321 ms; what that does to the lane's standing against
  `torch` has to be re-measured by the h2h harness, and the backward is only part
  of the op.

## Ledger effect

`max_pool3d` was last measured 4.66x slower than PyTorch (2026-08-08, interleaved
arms with A/A veto). Its backward's raw kernel just got ~4.2x cheaper. That is a
large fraction of the op, so the lane's standing should improve materially — but
the honest statement is that **it has not been re-measured against torch yet**, and
nobody should quote a new vs-upstream number until it is.
