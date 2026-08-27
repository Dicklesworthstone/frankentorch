# The SVD reduction is MEMORY-bound and the sweep is COMPUTE-bound. That determines the lever class for what is left.

Closes the "why" behind three separate rejects on this op — `wfiip`'s dispatch <5%, the gate
reject in `a01de6d8`, and the reduction's 0.98x thread scaling — with two counters on one run
instead of another stopwatch.

## The contrast

`perf record` on the same binary, same fixture (`FT_FIXTURE=generic`), n=512, one thread:

| symbol | instructions | cache-misses | misses per instruction |
|---|---|---|---|
| replay (`svd_bidiag_qr_f64` closure) | **63.18%** | **6.39%** | 0.10x |
| `matrixmultiply::dgemm_kernel` | 13.32% | **29.80%** | 2.2x |
| `bidiag::dot_rows_into_f64` | 5.96% | **22.68%** | **3.8x** |
| `bidiag::reduce_scaled_rows_f64` | 2.75% | **11.89%** | **4.3x** |
| `bidiag::bidiag_blocked_f64` | 5.03% | 6.10% | 1.2x |

The replay is 63% of the instructions and 6% of the misses. The reduction's BLAS-2 kernels are
about 9% of instructions and 35% of misses. They miss roughly **4x more per instruction than
average and ~40x more than the replay**.

## Why that explains everything already measured

**The sweep parallelises (3.86x from 1 to 8 threads).** It is compute-dense and cache-resident,
so it scales with cores, and reducing its instruction count converted almost exactly into time
(1.767x fewer instructions -> 1.75x faster phase, `ceb8da3d`).

**The reduction does not parallelise, and gets WORSE when forced to.** 143.977 ms at 1 thread
against 147.457 ms at 8. Forcing the parallel branch by lowering the gate made the phase 34%
slower (`a01de6d8`). A memory-bound phase gains no bandwidth from more threads; it only pays the
dispatch and the contention. That is not a gate that is mis-set, it is a phase that is not
thread-limited.

**The reduction runs at low IPC.** 13.7% of instructions but 28.2% of the 1-thread lane time.
Roughly half the IPC of the rest of the op, which is what a 4x miss rate buys.

## The lever class, stated so the next attempt does not repeat these three

For the REDUCTION, instruction count is not the constraint and neither is thread count. Both
have now been measured shut. The lever is DATA MOVEMENT: reuse, blocking, and layout. Anything
that reduces bytes moved per useful flop is on the table; anything that reduces instructions or
adds threads is not.

For the SWEEP, the opposite holds: it is compute-bound and cache-friendly, instruction count IS
the constraint there, and that is why the transposed replay converted.

## A corroborating detail worth chasing separately

`dgemm` is 29.80% of the misses on 13.32% of the instructions — the single largest miss
contributor. `project_blocked_kernel_staging_cost` already records that `gemm::dgemm` has no
`ld` parameter, so every blocked update stages packed copies, measured at 29.9% of the geqrf
lane at n=1024. This is an independent instrument pointing at the same staging cost, on a
different op.

## NOT claimed

Host load was 82.7 during the miss profile, and a shared LLC means ABSOLUTE miss counts are
inflated by peer processes. The claim here is the RELATIVE attribution across symbols within one
process, which contention distorts far less than it distorts wall time. No wall-time ratio is
asserted; the timing figures quoted are from the certified rows already banked in `38c31ae7`,
`ceb8da3d` and `a01de6d8`.
