# The degenerate SVD fixture distorts the FT/PyTorch RATIO itself: 1.021x on it, 2.341x on a generic matrix

Follow-up to `gqmws` (fixture hides the QR sweep) and `76c4bdea` (spectrum measured). That work
showed the default fixture cannot see one PHASE. This shows it also misprices the COMPARISON.

Retired instructions, so this is valid on a host at loadavg 125 with peer torch benchmarks
running — deterministic counts do not care. Estimator is the difference `(I_4 - I_1)/3`, which
cancels startup (`import torch` alone is larger than an n=512 factorisation). One thread each
side. Both fixtures reproduced BIT-FOR-BIT from the harness on both arms, so the only variable
is the matrix.

## Per SVD, n=512

| fixture | FT | PyTorch 2.12.1+cpu | FT/PT |
|---|---|---|---|
| `_mk` (default; 495 of 512 singular values equal) | 5.688e8 | 5.574e8 | **1.021x** |
| generic (512 of 512 distinct, cond 97.4) | 1.670e9 | 7.135e8 | **2.341x** |

**On the default fixture FrankenTorch and LAPACK execute essentially the SAME number of
instructions.** The loss the banked wall-clock rows record on that input (2.40x at n=512) is
therefore NOT extra work — at instruction parity, it is memory behaviour and IPC.

## The asymmetry is the mechanism

Going from the degenerate fixture to a generic one:

    FT work    5.688e8 -> 1.670e9    x2.936
    PT work    5.574e8 -> 7.135e8    x1.280

The degeneracy removes **66% of our work and only 22% of PyTorch's**. That is not a coincidence
of this matrix: the phase that deflates away is the bidiagonal QR sweep, which our
Golub-Reinsch path spends most of its instructions in and which `gesdd`'s divide-and-conquer
leans on far less. A fixture that deflates instantly is therefore differentially generous TO US,
and any ratio measured on it flatters FrankenTorch by roughly 2.3x in instruction terms.

## "Generic" is not a single number, and I am not pretending it is

Three fixtures, three answers, all on the same binary:

    _mk (3*I + low rank, 495-fold degenerate)     1.021x
    harness generic (+16 diagonal, cond 97.4)     2.341x
    fixed-LCG random (+4 diagonal)                3.94x   (znver3 build, earlier rows)

The ratio is monotone in how much QR-sweep work the spectrum demands. The honest statement is
NOT "the true ratio is X" but "the ratio is spectrum-dependent, the default fixture sits at the
extreme end where we look best, and a representative figure needs a stated spectrum".

## What this does and does not overturn

**Not overturned:** every previously banked FT/PT ratio remains a correct measurement OF ITS
INPUT. Both arms factored the same matrix; nothing was mis-measured.

**Overturned:** reading those ratios as "the SVD loss". They are the loss on a matrix whose
spectrum removes two thirds of our work and a fifth of the incumbent's.

**Consequence for prioritisation:** SVD has been ranked against other ops using numbers from
this lane. If the instruction gap is 1.02x on the bench input and 2.3x on a representative one,
the op's standing in the ledger is a function of the fixture, and any cross-op ranking that used
it should be re-read with that in mind.

## NOT claimed

No wall-time figure. Load was 125 with peer torch benchmarks throughout, so no ratio here is a
stopwatch claim and none is certified — these are instruction counts only. Whether the 2.341x
instruction gap on the generic fixture corresponds to a larger or smaller WALL gap than the
banked 2.40x is unmeasured, and needs a quiet window with `FT_FIXTURE=generic` and a real A/A
null (`FT_GATE_VALUES=<v>,<v>`).
