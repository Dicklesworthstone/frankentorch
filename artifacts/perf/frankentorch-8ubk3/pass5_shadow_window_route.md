# frankentorch-8ubk3 Pass 5 Shadow Active-Window Route

Date: 2026-06-12

Scope: deeper route after the exact-shift index-hoist source lever regressed.
No production source was edited in this pass.

## Trigger

Pass 4 preserved behavior but regressed the primary same-worker row:

```text
hz1 baseline eigvals_f64_256x256  [33.839 ms 34.014 ms 34.212 ms]
hz1 candidate eigvals_f64_256x256 [39.920 ms 41.337 ms 42.774 ms]
median ratio: 0.82x
```

The failure means index-hoisting and branch specialization are the wrong level
of attack for this residual. Do not repeat index arithmetic, branch splitting,
row-range, or alternate-shift micro-levers.

## Deeper Primitive

Next primitive:

```text
shadow active-window blocked Francis sweep kernel
```

Core idea:

1. Keep the current scalar shift source.
2. Keep the current selected-`m` search.
3. For one active Hessenberg window, clone the live window into scratch.
4. Build an ordered reflector/update ledger from the scalar sequence.
5. Apply the ledger to the scratch window with a blocked/tiled row/column update.
6. Compare the scratch window, shift stream, selected-`m` stream, deflation
   counters, and strict golden output against the scalar path.
7. Only after exact equality is proven should guarded production dispatch run
   the blocked window path.

This changes the implementation model rather than tuning scalar index math, but
it still honors the `frankentorch-8ubk3` lesson: shift policy is untouchable.

## Proof Obligations

The successor source pass must prove:

- identical `EigFrancisShiftSample` stream
- identical selected-`m` stream
- identical active-window stream
- identical deflation counters and sweep counts
- identical n=64/n=128/n=256 strict golden stdout SHA
- identical complex-pair slot ordering and sign convention
- no RNG and no tolerance expansion
- scalar fallback on any unsupported window shape or equality failure

## First Source Slice

The first acceptable source slice is not a public dispatch change. It should be
a private shadow-window proof harness that:

- runs scalar and blocked-window scratch paths from the same active-window state
- returns a compact equality report
- is callable from a focused test or diagnostic example
- exits before production dispatch if any byte-level or strict digest check fails

Only the next slice after that may attempt a production guarded dispatch.

## Pre-Score

```text
Impact 4: pass-1 profile still shows 319 n=256 sweeps and 1132 n=1024 sweeps.
Confidence 3: shift policy is preserved and scalar fallback is explicit.
Effort 5: scratch-window harness plus equality proof before a real speed path.
Score = 4 * 3 / 5 = 2.4
```

## Verdict

Productive route pass. Close `frankentorch-8ubk3` as rejected/rerouted and open
a successor bead for the shadow active-window blocked sweep primitive.
