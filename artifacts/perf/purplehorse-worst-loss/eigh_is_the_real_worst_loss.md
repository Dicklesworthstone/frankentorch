# `eigh` at n=512 is 5.60x SLOWER — the real worst loss, and it corrects my own headline

**Result: single-matrix `linalg.eigh` at n=512 is 5.599–5.628x SLOWER than PyTorch
2.12.1+cpu, CERTIFIED (A/A null 1.011, parity 6.12e-16 MATCH). That is nearly twice
the SVD square forward's certified 3.10–3.12x at n=1024, which I have called "the
worst loss in the tree" all session. It is not. `eigh` is.**

I found this by building the instrument whose absence I reported in `4d979a8c`, rather
than leaving the claim standing on a ledger.

## Why the claim was wrong, and why it was findable

Every standing this session was measured against a torch **co-process inside the same
invocation**, because this host has moved the incumbent 1.94x between two runs of the
same ELF. The eig/qr family had **no such harness at all** — `linalg_gap_sweep` is
FT-internal and its own header says torch is run separately. So "SVD is the worst op"
could only honestly be claimed as *"the worst op among ops that have a live-torch
harness"*, which is how `4d979a8c` phrased it. This closes that gap and the narrowed
claim turns out to have been hiding a bigger loss.

`bidiag_gate_sweep_h2h` now takes `FT_OP={svd|svdvals|eigh|eigvalsh|qr}`, driving both
arms and reusing the machinery already there: provenance, the A/A null via repeated
arms, per-round interleaving, the parity checksum.

Two details decide whether the rows mean anything:

* **Symmetrisation is not optional.** `torch.linalg.eigh` reads one triangle, and the
  existing fixture is non-symmetric. Unsymmetrised, the two arms would compute the
  spectrum of **different matrices** and the parity checksum would compare two correct
  answers to two different questions — agreement that means nothing. Both arms now
  apply `(A + Aᵀ)/2` in the same order. The resulting **6.12e-16** parity is the
  evidence it worked.
* **QR returns `|diag(R)|`** on both sides: R's diagonal is sign-ambiguous between
  implementations, its magnitudes are not.

## The measurement

```
FT_OP=eigh FT_ROUNDS=45 RAYON_NUM_THREADS=8 FT_GATE_SIZES="512" \
FT_GATE_VALUES="262144,262144" \
PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python \
  <snapshot of target/release/examples/bidiag_gate_sweep_h2h>
```

`elf_sha256=9e98e2eb1f7676c41a5eb40c13f8e05baeceaffbde75aca6a4c92e4c0eede73e`,
incumbent PyTorch 2.12.1+cpu threads=8 co-process in the same invocation, idle
**91.49% then 92.47%** before launch, loadavg 12.64 → 12.50, iowait 10.

| n | FT min (arm0 / arm1) | PT min | standing | **A/A null** | parity |
|---|---|---|---|---|---|
| 512 | 64.317 / 63.487 ms | 10.240 ms | **5.628x / 5.599x SLOWER** | **1.011 — PASS** | rel 6.12e-16 MATCH |

Earlier 15-round pass, same ELF, different window: 5.298x / 5.132x with null **1.022**.
The null **closed** with rounds (1.022 → 1.011), which is what sampling noise does —
unlike `conv2d_f32_masked`'s incumbent null, which sat at 1.041 across 16/32/64 and
never moved. And n=256 in that pass read 2.401x / 2.346x with its null already
**passing at 1.014**.

## What this does NOT contradict — I checked before claiming it did

I was about to write "this refutes the ledger's eigh figures." It does not.

The ledger records **eigh 4.84–7.92x FASTER** than torch, and that stands: those wins
are **batched tiny matrices** (B=2000, n=32–96), where torch's CPU batched
factorization loops serially over the batch and we parallelize per-plane. That was
structurally proven, not a thread artefact — torch `svdvals [2000,64]` reads 262/299/330 ms
at 1/8/32 threads, i.e. *slower* with more cores.

This measurement is a **single n=512 matrix**. Different regime entirely. Both are
true, and the batched vein remains harvested.

The ledger is in fact already consistent with what I measured, in the single-matrix
regime: *"eigh/svd reductions bandwidth-bound"*, with a single-matrix stage profile at
n=1024 of **reduce 444 ms / form-Q 180 ms / tql2-replay 1698 ms = 73%**.

## Where the time is, and why this is a different target from the SVD

That stage profile says the dominant single-matrix `eigh` cost is the **tql2 deferred-QL
replay at 73%** — *not* the reduction. The SVD's loss is the opposite shape: its
reduction is ~70% and its expansion is 33.5% (`c4d611c4`). So the two worst losses in
the tree do not share a target, and the SVD work does not transfer.

BlackThrush already shipped a row-blocked Givens replay worth 2.31–3.59x on this exact
phase (`76993cd1`), after finding the replay was **bandwidth-bound** rather than
compute-bound: the deferred whole-stream replay logs ~2n² Givens ops and then
`z.par_chunks_mut(n)` makes *every row* re-stream the whole ops vector from RAM. That
fix is in, and this 5.6x is what remains **after** it.

## A gate of mine that was vacuous here, stated plainly

My incumbent-plausibility check compared eigh's PT min (10.240 ms) against the **SVD's**
banked reference (~32.075 ms) and passed trivially, because there is no banked eigh
figure to check against. **That gate did not earn its pass on this run.**

Two independent reasons the incumbent is nonetheless credible: 10–11.5 ms is the right
order for LAPACK `dsyevd` at n=512 on 8 threads, and it reproduced across two windows
(11.542 and 10.240 ms). Our 64 ms is likewise consistent with `linalg_gap_sweep`'s
78.33 ms at 64 threads, 64t being slower than 8t on this op as established. Per-op
references should be added before more ops are banked through this gate.

`PT spread 2.13x` on the confirming row is inside my 3x ceiling but is the widest of
any row I have banked; the 15-round pass read 1.29x. Worth noting rather than burying.

## The board, corrected

| standing | ratio | null |
|---|---|---|
| **`eigh` n=512** | **5.60–5.63x SLOWER** | **1.011 CERTIFIED** |
| `conv2d_f32_masked_panel` (pre-fix route) | 4.49x | both PASS |
| SVD square forward n=1024 | 3.10–3.12x | 0.991 CERTIFIED |
| `conv2d_f32_masked` | 2.62–2.78x focused / 1.87x board | FT PASS |
| `eigh` n=256 | 2.35–2.40x | 1.014 PASS |
| SVD n=512 | 2.40x | 1.015 PASS |

Still unmeasured against a live incumbent: `eigvalsh`, `qr` (both now supported by
this harness and not yet run), plus `cholesky`, `lu`, `solve`, `inv`, `lstsq`, `pinv`,
`matrix_exp`, `det`, `matrix_rank`, non-symmetric `eig`. The ledger marks the direct
factorizations (`potrf`/`getrf`/`getri`) as MKL-batched walls; those are the least
likely to hide another surprise, and `eigvalsh`/`qr` are the most likely.
