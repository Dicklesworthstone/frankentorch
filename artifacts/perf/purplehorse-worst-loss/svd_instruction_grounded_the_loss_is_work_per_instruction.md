# SVD, measured DETERMINISTICALLY: we do 1.46x the FLOPs and 4.48x the INSTRUCTIONS. The loss is work-per-instruction.

Retired instructions and retired FLOPs do not depend on host load. That makes them
comparable ACROSS runs on a machine that has refused wall-clock window after window
(loadavg 15-99 for hours from peer criterion/PyTorch/callgrind), and it makes them
comparable against an A/A null that reads 1.049-1.203x on this very lane. Every number
below was taken while the host was busy, and none of them are affected by that.

## Method

`crates/ft-api/examples/svd_instr_count.rs` (new) and a mirrored Python driver, both
factoring the SAME matrix from the same fixed LCG (integer recurrence, power-of-two
scaling, `+4.0` diagonal bump), one thread each side, `torch.linalg.svd(full_matrices=False)`
against `tensor_linalg_svd(a, false)`.

**The estimator is a DIFFERENCE, not a total**: per-iteration = `(I_N - I_1)/(N-1)`. That
cancels process startup, allocator first-touch, and fixture construction. It matters
enormously here because `import torch` alone is ~6.4e9 instructions — larger than a whole
n=512 factorisation — so a raw total would have measured Python's startup and called it
PyTorch's SVD.

**One thread on each side, deliberately.** A retired-instruction count over a thread pool
includes the pool's spinning, which varies with load and would reintroduce exactly the
nondeterminism this method exists to escape. This measures WORK, not parallel efficiency.

**Fixture parity is PROVEN, not asserted.** Both arms print the same checksum to 13
significant figures:

    FT    n=512  iters=1  2.701351373108e1    iters=6  1.620810823865e2
    TORCH n=512  iters=1  2.701351373108e+01  iters=6  1.620810823865e+02
    FT    n=1024 iters=1  3.771935321319e1    TORCH n=1024 iters=1  3.771935321319e+01

Two independent implementations agreeing on sigma_max to 13 digits is what licenses the
comparison. Without it this would be two programs doing different work.

## The result (per SVD, n=512)

| quantity | FrankenTorch | PyTorch 2.12.1+cpu | ratio |
|---|---|---|---|
| retired instructions | 3.209e9 | 7.158e8 | **4.484x** |
| retired FLOPs (`fp_ret_sse_avx_ops.all`) | 1.842e9 | 1.263e9 | **1.458x** |
| FMA FLOPs (`.mac_flops`) | 6.633e8 | 9.663e8 | 0.686x |
| FMA share of FLOPs | 36.0% | 76.5% | |
| **FLOPs per instruction** | **0.574** | **1.765** | **0.325x** |
| IPC | 2.56-2.81 | 1.21-1.50 | |

n=1024: instructions 2.397e10 vs 4.751e9 = **5.046x**. The instruction gap GROWS with n,
in the same direction as the banked wall gap (2.40x -> 3.10x).

## What this establishes, and it redirects the campaign

**It is not an algorithmic-work problem.** We retire 1.46x the FLOPs. Some excess, not the
2.4-3.1x wall gap and nowhere near the 4.5x instruction gap.

**It is not a scheduling or IPC problem.** Our IPC is roughly TWICE PyTorch's (2.8 vs 1.5).
We issue efficiently; there is nothing to recover from stalls.

**It is a work-per-instruction problem.** PyTorch retires 1.765 FLOPs per instruction; we
retire 0.574 — below one, meaning the majority of our instructions are not arithmetic at
all (loads, addressing, loop overhead). MKL packs 3.1x more arithmetic into each
instruction, via wide vectors and FMA (76.5% of its FLOPs come from FMA, against our 36%).

**The wall gap UNDERSTATES the deficit**, which is the counterintuitive part: 4.48x the
instructions shows up as only 2.40x the wall time precisely because our better IPC absorbs
part of it. Anyone reasoning from the wall ratio alone is looking at a number that already
has a compensating effect folded into it.

**This is why every dispatch-level lever measured shut** (wfiip: reduction parallel dispatch
<5%; c4d611c4: removing the ENTIRE expansion phase still leaves 1.734x; 7ebc0555: the
form_p blocking gate a measured no-op). They all move where work runs or how it is split.
None of them changes FLOPs-per-instruction, which is the axis we actually lose on.

## The two causes are not equally available, and that is the useful part

1. **FMA contraction** turns `mul` + `add` into one instruction. It also CHANGES ROUNDING,
   and `dot_rows_into_f64` avoids it deliberately ("the product is formed into a temporary
   and added as a separate operation"). This one is blocked by the bit-exactness policy, not
   by implementation effort. It is a POLICY question and should be priced as one.
2. **Vector width** is bit-exact-safe in the across-rows form, because lane `j` accumulates
   row `j` alone and nothing is summed across lanes. The source records that `wide::f64x4`
   compiles to two `__m128d` on the SSE2 baseline; one 256-bit ymm would halve those FP
   instructions for identical results.

Cause 2 is the one available without a tolerance argument.

## NOT claimed

No wall-time ratio is asserted here; the 2.40x/3.10x figures are BANKED from earlier
certified runs and are quoted only for comparison, not re-measured. This method cannot see
parallel efficiency at all, by construction. And 1.46x the FLOPs is itself unexplained
excess worth its own look — it is simply much smaller than the instruction gap.
