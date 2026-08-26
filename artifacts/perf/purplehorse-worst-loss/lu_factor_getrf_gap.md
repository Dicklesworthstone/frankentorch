# `lu_factor` — bare getrf is 12.0x / 8.6-9.2x, and my "slogdet's residual IS the getrf gap" prediction MISSED

**Result: `lu_factor` measures 12.130x / 11.988x at n=512 (A/A null 1.000 / 1.003, both in
band — CERTIFIED) and 8.569x / 9.236x at n=1024 (null 1.000 / 1.028, arm1 outside the band
— measured, not certified). Parity 9.88e-13 and 7.64e-13 MATCH.
ELF `f255125c813109e1a73200ed3d8b3ce95343b36a994f87e7e49c821689d9038c`.**

Predicted ~10.8x at n=1024. Measured 8.569-9.236x on the paired estimator. **The prediction
missed by 1.17-1.26x.**

## What it was testing

After the slogdet double-factorisation fix, I claimed its residual 10.8-11.2x "IS
approximately our bare getrf gap". `lu_factor` is getrf with no tail on either side, and
its no-grad path already takes both outputs off one kernel call, so it tests that claim
directly.

## The claim is not settled, and the reason is instructive

| n=1024 | FT min | PT min | min/min | paired |
|---|---|---|---|---|
| post-fix `slogdet` | 39.634 ms | 4.520 ms | **8.77x** | 11.244x |
| `lu_factor` | 52.645 ms | 5.993 ms | **8.78x** | 8.569x |

On min/min the two ops are indistinguishable (8.77x vs 8.78x). On paired they differ by
1.3x. The two estimators give opposite answers to the question being asked.

**Torch itself ran 1.33x slower in the lu_factor window** (5.993 vs 4.520 ms), so the two
runs are different machine states and their absolute times are not comparable. This is the
host where an incumbent has moved 1.94x between two runs of the same ELF.

I have committed elsewhere in this campaign to trusting the paired per-round estimator over
min/min, on the grounds that a wide PT spread makes min/min compare our typical sample to
torch's luckiest escaped one. **Consistency requires applying that rule when it costs me**,
so: on the estimator I said I trust, the prediction missed. Switching to min/min here —
where it happens to vindicate me at 8.77 vs 8.78 — would be choosing the estimator after
seeing which one agrees.

**Neither estimator settles it.** A 1.2x question cannot be resolved across two runs on
this host at all. To determine whether slogdet carries anything above bare getrf, both ops
must be measured IN ONE INVOCATION against one live incumbent. That is the next
measurement, not a conclusion available now.

## One hypothesis of mine, raised and refuted within minutes

Seeing slogdet read worse than lu_factor on paired, I proposed that they use different
kernel entry points and that equating them was unjustified. **Checked, and wrong:**

```
crates/ft-kernel-cpu/src/lib.rs:25211  pub fn slogdet_contiguous_f64(...)
    line +27:  let factor = lu_factor_contiguous_f64(data, meta)?;
```

`slogdet_contiguous_f64` calls `lu_factor_contiguous_f64` directly. They share one LU core,
so slogdet is bare getrf plus an O(n) diagonal log-product and nothing else. That structure
predicts the min/min agreement (8.77 vs 8.78) and makes the paired 1.3x separation the
thing that needs explaining — most likely the two different machine windows rather than any
property of the ops.

## Standing

`lu_factor` at 12.0x (n=512, certified) and 8.6-9.2x (n=1024, measured) is a real row
either way, and places bare getrf in the same band as the other kernel-reaching
factorisations rather than with the 125-535x private-loop Householder primitives.
