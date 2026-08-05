# GroupNorm f32 — the "19x slower" loss does not reproduce

## Claim

**No loss exists.** f32 `group_norm` (no-grad) measures **2.94x FASTER** than
PyTorch, not 19x slower. No code change was made or needed; this entry exists to
retire the row from the loss list and to stop the next agent from re-attacking a
gap that is already closed.

## Measurement

Harness: `crates/ft-api/examples/groupnorm_h2h.rs` (pre-existing, median-ratio
with bootstrap CI). Local host, PyTorch CPU sidecar, 31 reps.

```
executing_elf_sha256=f8cfe5c867e61f1de2fcd660837df49e348f8042c4f6ec2ea4c2284242da4cc5
workload=group_norm_f32_no_affine [16,256,64,64] groups=32 reps=31
a_a_median_ratio=1.0186 ci95=[0.9489,1.0605] gate=PASS

op            FT(ms)    PT(ms)   verdict
  cat_anchor     10.251   42.378   FT 4.13x FASTER
  group_norm      3.677   10.821   FT 2.94x FASTER
```

Why this is trustworthy rather than another optimistic reading:

- **A/A null gate PASSES** at `1.0186`, `ci95=[0.9489, 1.0605]` — the host is
  quiet, so a 2.94x separation is not noise.
- **The `cat_anchor` calibrates the harness.** It is a previously-landed,
  independently-verified win, and it reads `4.13x FASTER` here. An anchor that
  still reports its known value is what distinguishes a real measurement from a
  bad window — the same discipline that caught the faked readings behind
  `frankentorch-1q8x` and `frankentorch-66pe`.
- **Executing-ELF SHA-256 is recorded**, so this is pinned to a specific binary.
- Parity checked first: 8 PyTorch probes within tolerance.

The in-tree levers the harness also re-verifies, both still earning their keep:
`scalar/SIMD = 5.78x` (`ci95=[5.5663, 6.0489]`, KEEP) and
`materialized/borrowed = 5.14x` (`ci95=[4.9759, 5.4850]`, KEEP).

## Why the old number said 19x

Not investigated in depth, but the shape of it is familiar. The f32 GroupNorm
surface has since acquired native f32 kernels (`group_norm_forward_f32`,
`group_norm_backward_f32`), an f32 grad fast path (`frankentorch-48w0b`), a SIMD
kernel, and a borrowed-input path — any of which postdates the row. The
"19x slower" figure is therefore best read as **stale**, superseded by landed
work, rather than as a live regression.

This is the second stale loss found in one session. The other was this repo's
headline SVD row (`FT 1874ms vs 9.9ms at N=256 = 189x`), which re-measured at
`11-15x` on a quiet worker with the PyTorch side unmoved — see
`artifacts/perf/frankentorch-svd-blocked-bidiag-r7jdo/svd_blocked_bidiag_ledger.md`.
Two independent stale rows is a pattern, not a coincidence.

## Recommendation

**Re-baseline the loss list before attacking any more of it.** The remaining
rows in circulation — BatchNorm2d 10x, linear 10.6-12.3x, conv2d 4-6x,
avg_pool1d 4-7x — have no h2h harness in `crates/ft-api/examples/` (only
`*_ab.rs` intra-repo probes and phase-timing probes exist for them), so none of
them has a calibrated, A/A-gated, anchor-checked number of the kind produced
above. They should each get one before any lever is chosen, because the cost of
optimizing a gap that does not exist is a whole session.

`cargo bench -p ft-api --bench pytorch_gauntlet_bench` is the canonical harness
and is the cheapest way to re-baseline them together.

## Retry predicate

Re-open the f32 GroupNorm row only if a calibrated h2h (A/A gate PASS, anchor
within its known value, ELF SHA recorded) shows FT slower on a shape that
matters. Do not re-open it on an uncalibrated bench reading.
