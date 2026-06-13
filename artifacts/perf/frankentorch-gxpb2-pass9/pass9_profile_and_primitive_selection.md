# frankentorch-gxpb2 pass 9 profile and primitive selection

Date: 2026-06-13
Agent: IvoryDeer
Bead: frankentorch-gxpb2

## Baseline

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench eigvals_f64_256x256 -- --warm-up-time 1 --measurement-time 3 --sample-size 10
```

Artifact:
`artifacts/perf/frankentorch-gxpb2-pass9/pass9_baseline_eigvals_256_vmi1227854.log`

Result:

```text
eigvals_f64_256x256     time:   [25.052 ms 25.747 ms 26.487 ms]
```

## Profile

Command:

```text
RCH_REQUIRE_REMOTE=1 RCH_WORKER=vmi1227854 rch exec -- cargo run --release -q -p ft-kernel-cpu --example eig_timing_probe
```

Artifact:
`artifacts/perf/frankentorch-gxpb2-pass9/pass9_eig_timing_probe_vmi1227854.log`

Rows:

```text
n=128   eigvals=    4.83ms  eig=    8.52ms  (vec_machinery=3.69ms)
profile n=128   sweeps=173 defl1=28 defl2=50 fallback=0 exceptional=0 max_width=128
n=256   eigvals=   28.03ms  eig=   50.27ms  (vec_machinery=22.24ms)
profile n=256   sweeps=319 defl1=14 defl2=121 fallback=0 exceptional=0 max_width=256
n=512   eigvals=  375.76ms  eig=  546.85ms  (vec_machinery=171.09ms)
profile n=512   sweeps=583 defl1=10 defl2=251 fallback=0 exceptional=0 max_width=512
n=1024  eigvals= 2939.18ms  eig= 5249.94ms  (vec_machinery=2310.77ms)
profile n=1024  sweeps=1132 defl1=18 defl2=503 fallback=0 exceptional=0 max_width=1024
```

The wall is still the scalar double-shift Francis QR sweep count. There are no
fallback or exceptional-shift events on the profiling matrix, so the next lever
must reduce normal sweeps or replace the sweep kernel. More row/column
micro-levers are explicitly rejected by pass 5, pass 7, and pass 8 evidence.

## Graveyard / Artifact Primitive

Canonical graveyard section used: `alien_cs_graveyard.md` section 9.6,
communication-avoiding algorithms. The relevant primitive is not TSQR itself;
it is the same data-movement doctrine applied to non-symmetric Schur reduction:
replace scalar BLAS-1 style bulge chasing with a blocked/small-bulge multishift
kernel and aggressive early deflation so updates can be accumulated and applied
as cache-friendly matrix blocks.

Prior fql10 child evidence constrains the route:

- AED suffix/whole-window threshold variants preserved goldens but failed
  same-worker gates.
- Direct two-bulge/four-shift public source was rejected because it cannot
  preserve the strict n=256 golden in one source lever.
- AED-derived alternate shift-list changed the strict n=256 digest.
- Exact row packing and far-row operation tape regressed.

## Ranking

1. **Size-gated large-n AED/multishift dispatch (`n >= 512`, values-only first).**
   - Impact: high on n512/n1024 (`375.76ms` and `2939.18ms` eigvals rows).
   - Confidence: medium-low because it changes floating-point order and cannot
     satisfy strict bit identity on the n256 golden path.
   - Effort: high.
   - Gate: strict `eigvals_golden` n64/n128/n256 SHA must stay unchanged by
     keeping the current scalar path for those sizes; large-n acceptance must use
     residual/eigenvalue-equivalence checks and same-worker n512/n1024 timing.
   - Score estimate: `2.0` only if the source lever is large-n gated and avoids
     any n256 strict-path drift.

2. **More proof-only AED records or hidden public diagnostics.**
   - Rejected. Pass 8 showed the linked diagnostic can perturb public timing and
     proof-only code has no public speed win.

3. **Further exact-preserving branch/range/packing edits in current scalar loop.**
   - Rejected. The family is exhausted by same-worker failures and does not
     reduce sweep count.

## Selected Next Lever

Do not edit public dispatch until coordinated with the fql10 parent owner. The
next source pass should be a single size-gated `n >= 512` values-only
AED/multishift candidate, with n64/n128/n256 strict golden fallback unchanged
and a separate large-n residual/eigenvalue proof. Target ratio: at least `2x` on
the n512/n1024 eigvals timing-probe rows, with no n256 golden drift.
