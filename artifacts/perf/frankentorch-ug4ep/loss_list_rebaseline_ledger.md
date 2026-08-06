# frankentorch-ug4ep - loss-list re-baseline vs PyTorch

## Claim

Of the five vs-PyTorch losses driving perf priority, **two are stale (one is now
a win), two are confirmed but smaller than listed, and one cannot be measured at
all** because no harness covers it.

The top perf target is no longer GroupNorm or linear. It is **`avg_pool1d` grad,
~5.0x slower**, which is the only row that reproduced at roughly its listed size.

## Result

Harness: `cargo bench -p ft-api --bench pytorch_gauntlet_bench` — the canonical
one, which runs the FrankenTorch and PyTorch arms inside the same Criterion
group against `pytorch_2_12_cpu`. Local host, PyTorch CPU sidecar.

| row | listed | measured | verdict |
|---|---|---|---|
| `avg_pool1d` grad | 4-7x slower | FT 32.283 ms vs PT 6.515 ms = **4.96x slower** | **CONFIRMED** |
| BatchNorm2d f32 grad | 10x slower | FT 29.357 ms vs PT 7.831 ms = **3.75x slower** | **CONFIRMED, smaller** |
| linear train hidden 2048 | 10.6-12.3x slower | FT 7.131 ms vs PT 16.328 ms = **2.29x FASTER** | **STALE — now a win** |
| GroupNorm f32 | 19x slower | FT 3.677 ms vs PT 10.821 ms = **2.94x FASTER** | **STALE — now a win** |
| conv2d | 4-6x slower | — | **UNVERIFIABLE, no harness** |

## Robustness of each call

Criterion ran 10 samples per arm and the PyTorch arms are wide, so every verdict
above is stated against the **worst case for FrankenTorch** — the PyTorch bound
that flatters PyTorch most:

- `avg_pool1d`: PT CI `[5.628, 7.749]`. Even at PT's *slowest* bound, FT is
  `32.283 / 7.749 = 4.2x` slower. The loss is real at any point in the interval.
- BatchNorm2d: PT CI `[6.750, 8.914]`. Even at PT's slowest bound, FT is
  `29.357 / 8.914 = 3.3x` slower. Real at any point in the interval.
- linear: PT CI `[12.072, 20.270]` — the noisiest arm here. Even at PT's
  *fastest* bound, FT at 7.131 ms is still `1.69x` faster. The win holds at the
  pessimistic end, which is the only reason it is claimed.
- GroupNorm: measured separately on a calibrated h2h harness with an A/A null
  gate (PASS, `1.0186`, `ci95=[0.9489, 1.0605]`), a landed-win anchor reading its
  known value, and the executing-ELF SHA recorded. See
  `artifacts/perf/frankentorch-groupnorm-f32-rebaseline/`.

BatchNorm2d additionally shows why the arm matters: the group contains two
FrankenTorch arms, `kgs4_114` at 50.137 ms and `kgs4_136_scalar_sum` at
29.357 ms. Quoting the wrong one turns 3.75x into 6.4x. The listed "10x"
predates both.

## conv2d has no harness

`pytorch_gauntlet_bench` has groups for `avg_pool1d`, `avg_pool2d`,
`batch_norm2d_f32`, **`conv3d`**, `linear`, `max_pool1d`, `max_pool3d`, and
`sdpa` — but none for conv2d. `crates/ft-api/examples/` has only intra-repo
probes for it (`conv2d_f32_grad_ab.rs`, `conv2d_f32_hessian_probe.rs`,
`conv2d_gradpenalty_probe.rs`, `conv2d_gradpenalty_probe.rs`), which compare
FrankenTorch against FrankenTorch and can say nothing about a PyTorch gap.

So the "conv2d 4-6x slower" row has **no traceable evidence in the tree**. It
should not drive priority until a `gauntlet_conv2d_grad` group exists. That is
the remaining scope of this bead.

## Why this mattered

Three of the five rows would have sent an agent to optimize something that is
already faster than PyTorch or already largely fixed. Two rows checked earlier in
the session had the same defect for a different reason — the SVD headline
(`189x`) was contention-inflated and re-measured at `11-15x` with the PyTorch
side unmoved (see
`artifacts/perf/frankentorch-svd-blocked-bidiag-r7jdo/svd_blocked_bidiag_ledger.md`).

The general lesson is cheap to state and expensive to skip: **a loss row is a
measurement, and measurements expire.** Re-measure before choosing a lever, and
quote the arm and the CI bound, not the headline.

## Next target

`avg_pool1d` grad, ~5.0x slower — now the largest confirmed vs-PyTorch gap in
the tree. FT arm is `frankentorch_kgs4_134_fused_sum_loss` at 32.283 ms; there
is already a phase-timing probe at
`crates/ft-api/examples/avgpool1d_phase_timing.rs` to start from.

## Follow-up: the avg_pool1d gap is mostly NOT avg_pool1d

Root-causing the one confirmed row before choosing a lever, via the existing
`crates/ft-api/examples/avgpool1d_phase_timing.rs`. Per-iteration split of the
`[8,64,8192]` f64 train step:

| phase | per-iter | share |
|---|---|---|
| `tensor_variable` (materialise input) | **22 677 us** | **43%** |
| forward | 13 247 us | 25% |
| backward | 9 566 us | 18% |
| sum | 474 us | 1% |
| session_new | 0.6 us | ~0% |
| TOTAL | 53 081 us | |

The gauntlet's FT arm builds its input INSIDE `b.iter` — `values.clone()` plus
`session.tensor_variable(...)` — so tensor materialisation is inside the measured
region. For the fused arm the bench actually timed (32.283 ms), `tensor_variable`
is roughly **70%** of the number, and the pooling work is roughly 30%.

**This is not automatically unfair**: the PyTorch script
(`benches/pytorch_avg_pool1d_grad.py`) also rebuilds its input inside its loop
via `base.detach().clone().requires_grad_(True)`. Both sides pay for setup. The
point is what the setup COSTS on each side. Splitting PyTorch the same way, same
34 MB, `torch.set_num_threads(8)`:

| | FrankenTorch | PyTorch | ratio |
|---|---|---|---|
| materialise input | 22 677 us | 6 022 us | **~3.8x slower** |
| pool fwd + sum + bwd | ~9 600 us (fused arm) | 6 960 us | **~1.4x slower** |

Caveat, stated because it bounds the claim: PyTorch was pinned to 8 threads here
while the FT phase probe used the box default, and the gauntlet gives PyTorch 32
threads (`FT_TORCH_THREADS` defaults to `32`), which is why its in-bench total
(6.5 ms) is below the 13.0 ms measured at 8 threads. So treat these as
indicative shares, not certified ratios.

The conclusion survives that caveat comfortably: **the avg_pool1d row is
dominated by input materialisation, and the pooling kernel itself is close to
parity.** Optimizing `avg_pool1d` would move ~30% of the number; optimizing
`tensor_variable` would move ~70% of it.

`tensor_variable` is not an avg_pool1d cost. Every gauntlet lane that builds its
input inside `b.iter` pays it — `avg_pool2d` and `batch_norm2d` are written the
same way — so this is a shared term sitting inside several "op losses",
including the BatchNorm2d row confirmed above. That makes it the higher-leverage
target of the two, and it means the remaining confirmed op-level gaps are
smaller than the gauntlet's headline numbers suggest.

At 34 MB in 22.7 ms, `tensor_variable` sustains ~1.5 GB/s, which is far below
memcpy speed for a copy of that size — consistent with an extra copy and/or a
serial first-touch fill, the same anti-pattern already fixed once in
`expand`/`broadcast_to` (`vec![v; numel]` serial first-touch, replaced by an
uninit buffer whose parallel fill does the first touch). Filed as its own bead.
