# RESOLVED: the SVD loss is ENTIRELY singular-vector accumulation. The values path is at instruction PARITY (1.126x).

This closes the contradiction banked in `e3cae389` and localises the worst loss in the tree to
one phase, deterministically, on a host too busy for any wall-clock claim.

## The contradiction, and why BOTH instruments were right

`perf` put 78.16% of retired instructions in `svd_bidiag_qr_f64` (the bidiagonal QR sweep); the
h2h harness reports that phase as `QR sweep 0.270 ms, 0%`. Both are correct — they measure
DIFFERENT OPERATIONS. The harness's banked phase split is the VALUES-ONLY path, where the sweep
genuinely costs nothing because the rotations are never accumulated into U and V. This driver
computes U and Vh, where accumulating them is the dominant cost.

The disassembly first ruled out the other candidate. The hot loop is a real Givens rotation, not
an LTO inlining artifact:

    vshufpd $0x1,%xmm2,%xmm2,%xmm3    swap [a,b] -> [b,a]
    vmulpd  %xmm2,%xmm1,%xmm2         [a*c, b*c]
    vmulpd  %xmm3,%xmm0,%xmm3         [b*s, a*s]
    vaddpd  %xmm3,%xmm2,%xmm4         [a*c+b*s, .]
    vsubpd  %xmm3,%xmm2,%xmm2         [., b*c-a*s]
    vmovsd  %xmm4,%xmm2,%xmm2         blend lane0 from add, lane1 from sub
    vmovupd %xmm2,(%rsi,%rdi,8)       store the rotated pair
    add     %r9,%rdi                  stride to the next pair

## The measurement (n=512, per SVD, one thread, same znver3 binary)

| | FrankenTorch | PyTorch 2.12.1+cpu | ratio |
|---|---|---|---|
| values-only (`svdvals`) | 3.721e8 | 3.306e8 | **1.126x** |
| full (U, S, Vh) | 2.817e9 | 7.158e8 | **3.94x** |
| vector accumulation (full - values) | 2.445e9 | 3.852e8 | **6.35x** |

(The 4.484x quoted earlier is the same comparison on a baseline build; `-C target-cpu=znver3`
takes it to 3.94x, bit-exactly — FLOPs and checksum unchanged.)

**Our bidiagonalisation-plus-values path is at instruction parity with LAPACK: 1.126x.** Every
instruction of the loss is in forming the singular vectors, where we spend 6.35x what PyTorch
does — a term that is by itself 3.4x PyTorch's ENTIRE SVD.

## Why this redirects the campaign

Years of levers here went at the REDUCTION and its dispatch. That phase is not the problem:
values-only, which is reduction + bidiagonal values, is at parity. It also explains the record
without contradiction — wfiip's <5% reduction dispatch, `c4d611c4`'s "removing the entire
expansion phase still leaves 1.734x", `7ebc0555`'s no-op gate, and `dot_rows_into_f64` (items
254, 258d, `4zjaa`) sitting at 3.26% of instructions. None of them touched vector accumulation.

## The mechanism, and what is available

We apply the Givens stream to U and V one rotation at a time: 128-bit `xmm` (two doubles), no
FMA, strided (`add %r9,%rdi`), and computing FOUR candidate values to keep TWO — the add/sub
pair is 50% discarded work. That is ~13 instructions for ~6 useful FLOPs, 0.46 FLOPs per
instruction, which is what drags the whole-op figure to 0.574. LAPACK's `gesdd` forms the
vectors by GEMM (BLAS-3) instead, which is why its density is 1.765.

Three observations, in increasing order of cost:

1. **The discarded lane is pure waste** and is bit-exact to remove: the two useful results are
   `a*c+b*s` and `b*c-a*s`, computed here as two full-width add and sub then blended.
2. **The rotation is 128-bit on a machine with 256-bit vectors.** Widening is blocked by the
   STRIDE, not by the arithmetic; the fix has the shape `4zjaa` already used for
   `dot_rows_into_f64` — SIMD across the strided dimension, each lane owning its own row so
   nothing is summed across lanes and bit-exactness is preserved.
3. **Deferring the stream and applying it as GEMM** is what `gesdd` does. The deferred-replay
   work already in the tree deferred the stream for FORK/JOIN GRANULARITY; it did not change
   the per-rotation instruction count. This is the big one and it is an algorithmic change.

## NOT claimed

No wall-time ratio. This is single-threaded and says nothing about parallel efficiency. It does
not establish that (1) or (2) converts into wall time — this host cannot currently certify that,
which is the entire reason for measuring instructions. And "at parity on values-only" is an
INSTRUCTION statement: the banked wall figures still have our values-only path slower than
PyTorch's full SVD, so that path's problem is density and memory behaviour, not instruction
count.
