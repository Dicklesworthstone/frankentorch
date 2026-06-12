# frankentorch-8ubk3 Pass 6 Closeout

Date: 2026-06-12

## Result

`frankentorch-8ubk3` is closed as rejected/rerouted.

Close reason:

```text
Rejected and rerouted: exact-shift index/branch source lever preserved golden
SHA but regressed same-worker hz1 eigvals_f64_256x256 from median 34.014 ms to
41.337 ms (0.82x). Source hunk removed; successor frankentorch-x9137 opened for
shadow active-window blocked Francis sweep proof harness.
```

## Final State

- Production source diff: none
- Kept source changes: none
- Strict golden SHA: `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`
- Final source attempt score: `0.0`
- Successor bead: `frankentorch-x9137`

## Successor

`frankentorch-x9137` targets the deeper primitive:

```text
shadow active-window blocked Francis sweep kernel
```

The first source slice must be a private proof harness:

- clone one active Hessenberg window into scratch
- consume the existing scalar shift and selected-`m` sequence
- apply a blocked/tiled row-column update ledger in shadow
- compare exact window, shift stream, selected-`m`, deflation counters, complex
  slot ordering, RNG absence, and strict golden SHA
- fall back to the current scalar path on any mismatch

No public dispatch should change until that exact proof passes.

## Handoff

Do not repeat:

- alternate shift packets
- AED replacement shifts
- range/index micro-cuts
- branch-specialization/index-hoist-only hunk

Start the successor from the pass-1/pass-4 evidence in
`artifacts/perf/frankentorch-8ubk3/`.
