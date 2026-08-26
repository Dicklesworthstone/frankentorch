# Why AVX2 barely helps the SVD reduction — the mechanism item 260c asked for

**Result: the row-dot is ALREADY hand-vectorised, `x86-64-v3` halves its packed
instruction count exactly as 256-bit-vs-128-bit predicts (6 packed ops → 3), and
NEITHER build emits a single FMA — by deliberate design, because contraction would
change the rounding. That is the mechanism behind the ~1.07x the flag measured, and it
is an answer rather than two more stopwatches.**

Item 260c asked for exactly this and said why:

> if it does not reproduce znver3's gain, a disassembly diff between the two to say
> WHY, because "AVX2 helps but only with Zen-3 tuning" is a claim that needs a
> mechanism, not two stopwatches.

## A correction I owe first

I set out to answer item 258d's question — *does LLVM form the SIMD-across-rows
pattern on its own?* — and my first instruction census came back "10 `mulsd` + 10
`addsd`, no packed ops", which I was about to report as "LLVM will not vectorise this
loop."

**That was my grep, not the compiler.** My pattern listed `vmulpd`/`vaddpd` but not the
non-VEX `mulpd`/`addpd`, so it silently dropped every packed instruction in the SSE2
baseline. A complete census shows 6 `mulpd` + 6 `addpd` sitting right there.

Worse, the question was already answered **in the source, at the site**: a prior
session ran this disassembly, recorded "10 `mulsd` + 10 `addsd` … and no `vfmadd` at
all", and *implemented* the SIMD-across-rows form in response, using `wide::f64x4`
gated on `rowdot_blocked()`. My baseline scalar counts reproduce that comment's
numbers exactly. The finding I thought I was making was already in the tree.

## The census

`ft_kernel_cpu::bidiag::dot_rows_into_f64`, a standalone symbol in both ELFs.

| instruction | baseline (`5650da0f…`) | **v3 (`2bfcd944…`)** |
|---|---|---|
| scalar mul (`mulsd`/`vmulsd`) | 10 | 18 |
| scalar add (`addsd`/`vaddsd`) | 10 | 18 |
| **packed mul (`mulpd`/`vmulpd`)** | **6** | **3** |
| **packed add (`addpd`/`vaddpd`)** | **6** | **3** |
| **FMA (`vfmadd*`)** | **0** | **0** |

Sanity check that makes the FMA row conclusive rather than a grep artefact: the same
two ELFs contain **152 `fmadd` instructions each** elsewhere. FMA is being emitted in
this build. It is absent from this function specifically.

## What the numbers mean

**The packed count halving is the flag working, not failing.** `wide::f64x4` lowers to
2 × `__m128d` on the SSE2 baseline and 1 × `__m256d` under `x86-64-v3`. So the same
vector work should cost exactly half the packed instructions at 256 bits — predicted
2:1, observed **6:3**. The AVX2 flag is doing precisely what it should to this
function.

**The FMA absence is deliberate and correct.** The source states it: the product is
formed into a temporary and added separately, and Rust does not contract `a * b + c`
without explicit fast-math. Contraction would round once where the scalar form rounds
twice, which would break the bitwise-identical property the whole blocked form rests
on. **This is a case where the fast instruction is unavailable for a correctness
reason, not an oversight** — and no compiler flag can or should change it.

**So AVX2's ceiling on this function was always low.** It cannot introduce FMA (barred
by rounding), and it cannot widen further than the `f64x4` the code already asks for.
All it can do is halve the instruction count for work that was already vectorised —
which is real but bounded, and entirely consistent with the **~1.07x at n=512 and
nothing at n=256** measured in commit `636ce5bc`.

## What this closes, and what it does not

**Closes item 260c's "why".** "AVX2 helps but only with Zen-3 tuning" was the wrong
frame. The truth is narrower and more useful: on the reduction's hot loop AVX2 has
almost nothing left to win, because the loop is already hand-vectorised at `f64x4` and
the one big remaining instruction-level lever — FMA — is closed by the bit-exactness
requirement. A znver3-vs-v3 diff would be measuring scheduling around an
already-saturated form.

**Does not close the standing.** The SVD square forward at n=1024 is still
**3.10–3.12x SLOWER, certified** (`7cf74314`). This explains why one lever was small;
it does not find a new one.

**And it reframes item 258d's proposal as already-harvested.** That item asked whether
the SIMD-across-rows form should be written. It was written, it is in the shipped
binary, and its own author declined to claim a speedup for it: *"Whether that converts
into wall time is NOT claimed here: this lane's A/A null measured 1.049–1.203x on this
host, so an effect of this size is below what the instrument can resolve."* That
matches what I measured independently from the other direction — the reduction's
parallel dispatch is worth **<5%**, itself below the instrument's floor
(`95a70cd8`).

Two separate levers on this loop, both real, both below the noise floor of the only
instrument we have. That is the honest state of the SVD reduction: it is not obviously
mis-implemented, and the remaining gap against MKL is not going to be recovered by
instruction selection on this function.
