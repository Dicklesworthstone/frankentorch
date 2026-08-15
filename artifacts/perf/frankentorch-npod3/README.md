# npod3 — GroupNorm f32 train step, re-measured against a live arm

The scorecard row (`frankentorch-kgs4.115`) records this workload at **19.04x
slower** — FT 11.72 ms vs PyTorch 0.615446 ms, `[8,64,28,28]`, 32 groups, affine
grads, sum loss. That number predates `frankentorch-48w0b`, which added the f32
affine-grad fused path, so the first job was to find out what the standing IS.

## Provenance

- ELF SHA-256 `c7c0a3b2f8f6e59b02edd0804e1cd4d0ffb0083a752ea9a7a39c989325af355d`,
  self-reported from inside the process
- incumbent PyTorch **2.12.1+cpu**, self-reported by the arm, same invocation,
  torch threads 8
- AMD Threadripper PRO 5975WX, 64 threads observed, 1 NUMA node, governor
  `performance`, mimalloc (`--features fair-alloc`)
- 6 invocations, load average 14.7–29.4; raw in
  `artifacts/perf/frankentorch-npod3-groupnorm.txt`
- f32 on BOTH arms, affine parameters requiring grad on both, identical timed
  region (forward, loss_sum, backward), leaves built outside the timer

## Certified row

Balanced square, **both** arms' A/A gates PASS, parity `match` (run 6):

| lane | FT (ms) | PT (ms) | standing | null_PT | null_FT |
|---|---|---|---|---|---|
| `group_norm_f32_kernels` | 6.449 | 0.301 | **21.40x SLOWER** | PASS [0.823,1.113] | PASS [0.834,1.103] |

Session lane across all six runs, parity `match` every time but gates not yet
clean: **8.81x, 8.83x, 9.03x, 9.38x, 9.53x, 10.93x** (FT 3.17–3.86 ms vs PT
0.347–0.404 ms).

## The correction I owe my own earlier reading

The first (parity-invalid) run showed the raw-kernel lane *slower* than the
session lane, and I read that as "the engine is not the term, the kernel is".
**That inference was wrong, because the two lanes do not take the same kernel
route.**

- the session lane hits the sum-loss shortcut — `group_norm_f32_sum_shortcuts`
  in ft-api, `group_norm_sum_forward_f32`, and `group_norm_backward_scalar_f32`,
  which never materializes an upstream gradient at all
- the `_kernels` lane calls the GENERAL `group_norm_forward_f32` +
  `group_norm_backward_f32`, and additionally allocates and fills a 1.6 MB
  all-ones `dy` inside the timed region, which the general backward then scans
  end-to-end (`dy.par_iter().all(...)`) purely to discover it could have used the
  scalar path

So the split as built measures **general route vs shortcut route**, not **kernel
vs engine**. It is still a real and useful number — it prices the route a caller
who does not use a sum loss actually gets — but it does not isolate the engine
term, and no conclusion about the f64 grad-space round trip can be drawn from it.
Isolating that needs a third lane calling the same shortcut kernels the session
uses, with no session around them.

## What is settled

1. **The 19.04x scorecard row is stale for the session path.** The workload it
   names now runs at **~9x** (8.81–10.93 across six invocations, parity match).
   Still a large loss, and still the campaign's biggest normalization gap, but
   not 19x, and the scorecard should be corrected rather than re-quoted.
2. **The general (non-sum-loss) route is worse than the headline** — 21.40x,
   certified. Any caller whose loss is not a plain sum gets that one.
3. The 19.04x figure is closest to what the GENERAL route measures today, which
   is consistent with the shortcut having been added since the row was recorded.

## What is NOT established

- Where the ~9x sits inside the session path. The engine/conversion term is
  UNMEASURED — see the correction above.
- Whether the kernels are bandwidth-bound. 401,408 f32 is 1.6 MiB; at 3.2 ms the
  session lane moves it at well under 1 GB/s, which does not look bandwidth-bound
  at all, but that is arithmetic, not a measurement.
