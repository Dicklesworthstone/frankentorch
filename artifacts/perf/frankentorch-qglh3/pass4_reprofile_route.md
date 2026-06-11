# frankentorch-qglh3 pass 4 reprofile route

Date: 2026-06-11

## Context

Pass 3 rejected the values-only AED suffix deflation. The final exact-source
same-worker retry on `vmi1227854` was `[28.131 ms 29.767 ms 31.029 ms]` against
the pass-1 baseline `[26.514 ms 27.445 ms 29.029 ms]`.

The rejected hunk never addressed full `eig` because it was values-only. The next
route must therefore attack the real qglh3 scope: full AED data plumbing and
multishift shift handoff.

## Post-Rejection Probe

Command:

```bash
env RCH_REQUIRE_REMOTE=1 rch exec -- cargo bench -j 1 -p ft-kernel-cpu --bench linalg_bench -- eig_f64_256x256 --sample-size 10 --warm-up-time 1 --measurement-time 3
```

RCH selected `vmi1227854`. The run completed while the fleet was degraded and the
same worker had other active jobs.

Result:

| Row | Estimate |
| --- | ---: |
| `eig_f64_256x256` | `[70.709 ms 80.543 ms 85.978 ms]` |

## Interpretation

This is routing evidence, not a regression claim:

- pass 3 is removed in the current source;
- the worker was under visible load;
- the pass-1 full `eig` comparator on the same worker was
  `[49.080 ms 49.892 ms 51.006 ms]`.

The useful conclusion is structural: qglh3 remains full AED work, not more
values-only threshold work. The next lever should create a proofable AED window
record with:

- `kw`, `en`, fixed `nw`;
- window Schur values;
- window Schur vectors `Z`;
- conservative deflation count;
- undeflated shifts in current public ordering.

Public full-`eig` wiring requires focused reconstruction/orthogonality tests and
a same-worker speedup gate.
