# The SVD h2h fixture has 495 of 512 singular values EQUAL. Every phase conclusion drawn from that lane is a fixture artifact.

`frankentorch-i040z` was closed as "QR vector replay is 0% of full-SVD time; evidence in
comment", on a live H2H whose METHOD is sound — passing A/A null (1.022), live PyTorch in one
invocation, parity MATCH, ELF sha, guard PASS. I am not disputing the measurement. I am
reporting what its input is, because the same timer reads ~95x larger on a generic matrix and
the difference is entirely the fixture.

## The fixture

`bidiag_gate_sweep_h2h`'s `_mk(n, False)`, used by the svd/svdvals/qr/geqrf/orgqr/ormqr lanes:

    A = ((((r + 2) * (c + 3)) % 17) - 8.0) * 0.05 + eye(n) * 3.0

`((r+2)*(c+3)) % 17` takes 17 values and is periodic in both indices, so A is `3*I` plus a
LOW-RANK perturbation. Measured spectrum at n=512 (`torch.linalg.svdvals`):

    top 6    : 55.2329 54.5643 50.6957 47.8738 42.2388 42.0704
    middle   : 3.0 3.0 3.0 3.0 3.0 3.0
    bottom 6 : 2.5787 2.2119 1.8964 1.4142 0.6493 0.1566
    exactly 3.0 (within 1e-9): 495 of 512
    distinct values (6dp)    : 18

## Why that specifically hides the QR sweep

The implicit-shift bidiagonal QR sweep terminates by DEFLATION: a superdiagonal entry that is
negligible against its neighbours splits the problem. When 495 of 512 singular values are
already identical, the bidiagonal form deflates almost immediately and the sweep performs
close to zero iterations. It is not that the sweep is cheap; it is that this input asks it to
do nothing.

## The same counter, the same build, a generic matrix

Read via the harness's own accessors (`svd_reduction_sweep_ns_take`,
`svd_deferred_left_phase_ns_take`) from a driver over a fixed-LCG random matrix, n=512:

    1 thread   reduction 42.995  form_pq 53.738  sweep 85.301   | dl_qr 182.047 gemm 8.227 assemble 4.300  hits=1
    8 threads  reduction 52.134  form_pq 52.730  sweep 24.291   | dl_qr 129.164 gemm 2.352 assemble 5.417  hits=1

The three phases sum EXACTLY to `dl_qr` (52.134 + 52.730 + 24.291 = 129.155 vs 129.164), so the
instrumentation is self-consistent and `hits=1` confirms the same deferred-left route the
harness takes. **Sweep is 24.291 ms at 8 threads here against 0.255 ms there — ~95x — on the
same code, same n, same thread count, same counter.** The only difference is the matrix.

This also retires the contradiction banked in `e3cae389`: the harness timer is NOT mis-scoped
(`SVD_SWEEP_NS` does wrap `svd_bidiag_qr_f64`, replay included) and the perf profile is not an
inlining artifact. Both instruments were right; the fixture differs.

## What this does and does NOT overturn

**Still valid:** the peer's FT/PT ratio on that input. Both arms factor the same matrix, so
2.465x/2.497x is a fair comparison OF THAT INPUT, and LAPACK gets the same deflation gift.

**Does not generalise:** any statement of the form "phase X is/is not the bottleneck" taken
from this lane. The banked SVD phase map (reduction 69-76%, form_p/q 24-30%, sweep 0%) is a
property of a spectrum with 495-fold degeneracy, not of the SVD.

**Inverts the usual hazard.** The standing rule is to reject a lever that only pays on the
bench input. Here the bench input is the unrepresentative one: it makes a real cost invisible.
A lever rejected on this fixture has not been priced on anything a user would factor.

## What I am NOT claiming

No wall-time claim for the transposed replay (`8e077e39`). Host load was 33-133 throughout, and
the numbers above are phase SHARES within single runs, quoted to separate two instruments by an
order of magnitude, not to certify a speedup. Whether 1.767x fewer instructions in the sweep
becomes wall time on a generic matrix is still unmeasured and still needs a quiet window.

## Recommendation

Add a generic-spectrum fixture to the SVD lane (the existing `_mk` can stay for the lanes that
want a cheap deterministic matrix) before any further SVD phase conclusion is drawn. Until then
the SVD lane cannot see the sweep at all.
