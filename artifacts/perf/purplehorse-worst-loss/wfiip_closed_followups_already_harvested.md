# wfiip: closing it — both owed follow-ups are already harvested, and my "stale bound" hypothesis is REFUTED

`frankentorch-svd-reduction-parallel-below-floor-wfiip` sat P1/OPEN as the last ready
CAPABILITY/PERF item. It is a COMPLETED measurement, not pending work: it establishes a
BOUND (the reduction's parallel dispatch is worth <5% at n=512) and says explicitly of
another rounds-doubling, "Do not run it." Verified before closing that the two things it
left owed are both landed.

## Owed item 1 — "the bit-exact form is SIMD ACROSS ROWS". SHIPPED.

`dot_rows_into_f64` (ft-kernel-cpu) already does exactly this, four rows at a time, and it
has been the default since item 254 (`ROWDOT_BLOCKED`, default `true`, with the pre-254
one-row loop retained as an in-process measurement arm so the A/B needs one binary and one
window rather than two).

## Owed item 2 — "whether LLVM already forms that is a question for a disassembly". ANSWERED.

Answered by a peer under `frankentorch-4zjaa`, and the result is recorded in the source:

    objdump of dot_rows_into_f64 in a built ELF gave 10 mulsd + 10 addsd against
    2 mulpd + 2 addpd, and no vfmadd at all — four scalar dependency chains.

LLVM did NOT form it. The four accumulators bought instruction-level parallelism and
nothing wider. The fix shipped in `a089a7b4`: explicit `wide::f64x4`, lane `j` holding row
`j`'s accumulator and adding only row `j`'s products in ascending `c`, from the same `0.0`.
Nothing is summed across lanes, which is what makes it bit-exact where vectorising a SINGLE
row's dot product would not be.

## The hypothesis I formed, and why it is WRONG

I noticed the source comment says `f64x4` is two `__m128d` on an SSE2 baseline, and that
wfiip cites an AVX2 bound of ~1.07x at n=512. Under `x86-64-v3` an `f64x4` maps to ONE ymm
instead of two xmm, so I hypothesised that the AVX2 figure was measured BEFORE the f64x4
form landed and was therefore a stale bound for today's code — i.e. AVX2 might now be worth
more than it was priced at.

REFUTED, by ordering:

    a089a7b4  2026-08-24  SIMD across rows in step-(12)        <- the f64x4 change
    636ce5bc  2026-08-26  x86-64-v3 does NOT reproduce ...     <- the AVX2 measurement
    git merge-base --is-ancestor a089a7b4 636ce5bc  -> TRUE
    git show 636ce5bc:crates/ft-kernel-cpu/src/lib.rs | grep -c "wide::f64x4"  -> 7

The AVX2 measurement was taken on a tree that already contained the f64x4 form. The ~1.07x
bound is CURRENT, not stale, and there is no re-measurement owed here. Recording the wrong
turn so the next reader does not take it: the date order is the whole answer, and it is
cheaper to check than to re-measure.

## Standing after this

The dispatch- and instruction-level search on the SVD reduction is closed. Per wfiip's own
summary, what remains needs the per-core matvec made genuinely faster in pure Rust, or
parallelism the current gate cannot express — not another gate or flag. No ready
CAPABILITY/PERF bead remains actionable.

## Not measured

Host load was 87-99 throughout this session's later windows with peer criterion/PyTorch and
callgrind running, so nothing here is a stopwatch claim. Every statement above is from the
source, the git history, or an objdump a peer recorded.
