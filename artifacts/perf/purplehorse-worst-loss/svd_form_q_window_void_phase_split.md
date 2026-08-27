# SVD after the blocked `form_q`: window VOID (no A/A null, load 30->128) — but the phase split stands

Run: `bidiag_gate_sweep_h2h`, `FT_OP=svd`, n=512/1024, 24 rounds, warmup 6, `RAYON_NUM_THREADS=8`,
ELF `dd5609c32a890c2c2a75dfcaccd42768b040c0a5e04d688c21adb71ba09313b1`,
incumbent PyTorch 2.12.1+cpu driven as a co-process in the SAME invocation.

## The vs-torch ratios from this run are NOT quotable. Two independent reasons.

**1. I did not run an A/A null.** The harness's null is "repeat an arm in `FT_GATE_VALUES`".
I accepted the default two arms, which are `262144/...` (gated) and `SERIAL/...` — those differ
by the lever under test, so the `paired-vs-arm0` column (1.033x @512, 1.055x @1024) is an **A/B,
not a null**. This run therefore has no noise floor, and per
`feedback_aa_null_is_the_noisy_part` an uncertified ratio here is worth nothing.
Correct invocation next time: `FT_GATE_VALUES=262144,262144`.

**2. The window drifted hard.** Load `30.56 -> 59.92` across n=512 and `59.92 -> 128.24` across
n=1024 — it more than quadrupled during the run. PT's own sample spread was **2.53x / 2.28x**
(GATE 2b discards above ~3x, so this passed only marginally), iowait climbed 26,910 -> 162,500
jiffies, and CPU clock swung 3785/2514 then 2394/4277 MHz. Post-run `ps` showed two peer `rustc`
at 179% and 102% plus a `python3` at 99.8%. This is exactly the contended window
`feedback_peer_bench_contention` says fakes results.

For the record and NOT as a claim: the run read 2.479x @512 and 3.987x @1024 against banked
2.40x / 3.10x. I am not asserting a regression — a drifting window inflates the FT arm, and
`feedback_aa_null_blind_to_scaled_incumbent` requires checking PT's absolute ms against a banked
figure before believing any direction. That check is still owed.

## What DOES survive: the phase split, and it is the useful finding

Phase attribution is FT-internal (median of 3 instrumented calls), so contention distorts
magnitudes but not the ratio between our own phases:

| n | reduction | form_p/q | QR sweep |
|---|---|---|---|
| 512  | 42.5-44.9 ms | **69-73%** | 16.6-18.7 ms | **27-30%** | 0.27 ms | 0% |
| 1024 | 321-346 ms   | **66-69%** | 156-165 ms   | **31-34%** | 0.36-0.46 ms | 0% |

**`bidiag_form_q_blocked_f64` targets the MINORITY phase.** form_p/q is 27-34% of the lane;
the reduction is 66-73%. This is the same shape as the eigh/`dstedc` error recorded in
`project_eigh_phase_map` — a named-algorithm lever aimed at the smallest phase — and it is
worth flagging before anyone extends the blocking work.

**Measured size of the lever, FT-vs-FT, n=1024:** gated 439.330 ms vs SERIAL 452.467 ms by min,
`paired-vs-arm0` 1.055x. Roughly 3-5%. That is consistent with
`frankentorch-svd-reduction-parallel-below-floor-wfiip` ("parallel dispatch is worth <5%") and
does not contradict the peer's correctness work.

**At n=512 the blocked path never ran.** Branch counters are `(0, 0, 0)` for both arms: the
default gate is 262144 = 512x512, and n=512 does not exceed it. Only n=1024 shows `(512, 0, 512)`.
So n=512 says nothing whatsoever about `form_q`, and any n=512 figure quoted for it is vacuous.

## Parity is clean

`rel 1.90e-12` @512 and `6.07e-14` @1024, MATCH on both arms — within the ratified
reconstruction/orthogonality policy for eigensolver/SVD vector outputs
(`project_eig_tolerance_policy_ratified`). The blocked `form_q` is not a correctness problem.

## What is owed

A re-run in a quiet window with `FT_GATE_VALUES=262144,262144` for a true A/A null, plus PT's
absolute ms checked against a banked figure. Until then the SVD vs-torch number stands at its
previously banked 2.40x / 3.10x, unrevised.
