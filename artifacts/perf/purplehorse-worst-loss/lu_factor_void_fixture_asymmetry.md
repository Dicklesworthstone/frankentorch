# `lu_factor` row VOID on parity — and the void row would have made me retract a correct claim

**Result: no ratio. The n=512 and n=1024 rows are void on a parity MISMATCH of 7.75e-1 and
8.40e-1. Cause was a fixture asymmetry in my own harness. Reporting the failure rather than
quietly re-running it, because of what the void row nearly cost.**

ELF `ae298a7280a57f08681b3039d0b1f4c7caa21e66713bc829ae8856583c15f848`

| n | FT min (arm0/arm1) | PT min | ratio READ | A/A null | spread gate | parity |
|---|---|---|---|---|---|---|
| 512 | 16.446 / 16.798 ms | 1.368 ms | 12.495x / 12.813x | 1.000 / 0.991 PASS | 1.56x ok | **7.75e-1 MISMATCH** |
| 1024 | 53.114 / 48.729 ms | 8.669 ms | 6.622x / 6.711x | 1.000 / 1.009 PASS | 1.23x ok | **8.40e-1 MISMATCH** |

## Cause

`lu_factor` was added to the incumbent's `_spd` fixture selector but NOT to the Rust arm's.
Our arm factorised the raw `fill(n)`; torch factorised `_spd(n)`. **Two different
matrices.** The Rust selector read

```rust
} else if ft_op == LinalgOp::Cholesky || ft_op == LinalgOp::Slogdet || ft_op == LinalgOp::Inv {
```

and `LuFactor` fell through to the default `fill(n)` branch.

## Why this is worth a write-up instead of a silent fix

**Every gate except parity passed.** Both A/A nulls landed in band, both incumbent-spread
gates passed, idle was 93.99%/94.32%, and the run was even discarded and retried once by
GATE 2b before producing these numbers. A row can clear every timing gate and still be
measuring two different computations.

**The void number was plausible, and it pointed the wrong way.** The n=1024 row read
6.622x against my predicted ~10.8x. That is not an absurd figure — it is exactly the shape
of an honest refutation, and I had publicly committed to the prediction beforehand. Without
the parity check I would have retracted the getrf claim on the strength of a fixture bug.
**A wrong retraction costs as much as a wrong assertion and is much harder to notice**,
because withdrawing your own claim feels like rigour.

This is the third time this campaign that a PASSING A/A null has certified something
meaningless. The null compares two FT arms inside one run; it proves they agree with each
other and can say nothing about whether either is computing the intended thing.

## Fix

`LuFactor` added to the Rust selector, plus a structural check that both fixture selectors
enumerate the same ops — the bug CLASS, not just this instance:

```
python SPD selector: ['cholesky', 'inv', 'lu_factor', 'slogdet']
rust   SPD selector: ['cholesky', 'inv', 'lu_factor', 'slogdet']
ASYMMETRIC: none — both sides agree
```

Any op present in one selector and absent from the other silently compares different
matrices while passing every timing gate. Re-measuring with matched fixtures.
