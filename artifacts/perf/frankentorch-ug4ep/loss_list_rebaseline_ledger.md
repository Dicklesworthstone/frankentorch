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

## Correction: it is the ALLOCATOR, not `tensor_variable`

The `tensor_variable` hypothesis above (and `frankentorch-uqsit`, filed on it) is
**refuted by reading the code**. `tensor_variable` -> `tape.leaf` ->
`DenseTensor::from_contiguous_values` -> `from_storage` -> 
`TensorStorage::F64(Arc::new(storage))`. That is a **move into an `Arc`**. It
performs no copy and there is nothing in it to optimize.

The 22 677 us attributed to "`tensor_variable`" is the probe's own
`base.clone()` in the same timed region — a fresh 32 MB `Vec<f64>` allocation
plus memcpy. Under glibc `malloc`, an allocation that size is served by `mmap`
and returned by `munmap` on drop, so **every iteration re-faults all 8192
pages**. That is the ~1.5 GB/s. PyTorch never pays it, because its caching
allocator hands back the same warm block each iteration.

So the asymmetry was real but it was never a tensor-code defect: it is
first-touch page-fault churn, and it lives in the harness's input rebuild, which
both sides perform.

The repo already anticipated this — `pytorch_gauntlet_bench` documents a
`fair-alloc` feature (mimalloc) "for allocator-sensitive FT/PyTorch
comparisons". Re-running the two confirmed rows under it:

| row | default allocator | **`--features fair-alloc`** |
|---|---|---|
| `avg_pool1d` grad | FT 32.283 vs PT 6.515 = **4.96x slower** | FT 11.560 vs PT 6.025 = **1.92x slower** |
| BatchNorm2d f32 grad | FT 29.357 vs PT 7.831 = **3.75x slower** | FT 7.408 vs PT 6.117 = **1.21x slower** |

Both PyTorch arms are unchanged within noise (6.5 -> 6.0, 7.8 -> 6.1), which is
the control: the allocator swap moved the FrankenTorch side and left PyTorch
alone, exactly as the mechanism predicts.

## Final re-baseline

| row | listed | true standing |
|---|---|---|
| GroupNorm f32 | 19x slower | **2.94x FASTER** |
| linear hidden 2048 | 10.6-12.3x slower | **2.29x FASTER** |
| BatchNorm2d f32 grad | 10x slower | **1.21x slower** (near parity) |
| `avg_pool1d` grad | 4-7x slower | **1.92x slower** |
| conv2d | 4-6x slower | **no harness — unverifiable** |

**Not one of the five rows is a large real loss.** Two are wins, two are within
2x under a fair allocator, and one has no evidence at all. The remaining
headroom on this list is roughly 2x on a single op, not 5-19x across five.

Anyone quoting these rows should quote the fair-alloc number, or state plainly
that the default-allocator number includes per-iteration `mmap`/`munmap` churn
that PyTorch's caching allocator avoids.

## Open question worth a decision, not a lever

Should FrankenTorch ship a caching allocator by default? PyTorch effectively
does. Today `fair-alloc` is opt-in and off in normal builds, so **real users get
the slow path on any workload that repeatedly allocates large buffers** — the
gauntlet is only unusual in making that visible. That is a product decision
about default behaviour, not a kernel optimization, so it is recorded here
rather than actioned.

## conv2d lane built — and the row is stale too

The missing harness now exists: `gauntlet_conv2d_grad` in
`crates/ft-api/benches/pytorch_gauntlet_bench.rs` plus
`benches/pytorch_conv2d_grad.py`. Shape `8x64x32x32` against a `64x64x3x3`
kernel, stride 1, padding 1, grads on both input and weight, `sum()` loss —
sized into the same per-iteration band as the conv3d lane so it fits Criterion's
3 s window.

**Parity verified first, because a timing lane is worthless if the two sides
compute different things** and the gauntlet groups do not assert that
themselves. `crates/ft-api/examples/conv2d_gauntlet_parity.rs` reproduces the
lane's exact workload; it agrees with the PyTorch script to all 12 printed
digits:

| | FrankenTorch | PyTorch |
|---|---|---|
| output shape | `[8, 64, 32, 32]` | `(8, 64, 32, 32)` |
| loss | `-1.142627639693` | `-1.142627639693` |
| `x.grad[0]` | `0.591702728516` | `0.591702728516` |
| `w.grad[0]` | `7.440962005517` | `7.440962005517` |

Result:

| allocator | FrankenTorch | PyTorch | standing |
|---|---|---|---|
| default | 23.519 ms | 29.814 ms | **1.27x FASTER** |
| `fair-alloc` | 24.148 ms | 30.988 ms | **1.28x FASTER** |

`conv2d` listed at "4-6x slower" is **FASTER**, by 1.27x. The call holds at the
bound that flatters PyTorch most: FT's slowest (23.779) against PyTorch's
fastest (27.734) is still 1.17x in FrankenTorch's favour.

Note the allocator barely moves this lane (1.27x vs 1.28x), unlike avg_pool1d
and BatchNorm2d. That is a consistency check on the allocator finding rather than
a contradiction: conv2d's input here is 4 MB against those lanes' 32 MB, and its
compute per iteration is far higher, so per-iteration `mmap` churn is a small
share of the number. The allocator effect appears exactly where the mechanism
predicts it should.

## Final answer for this bead

| row | listed | true standing |
|---|---|---|
| GroupNorm f32 | 19x slower | **2.94x FASTER** |
| linear hidden 2048 | 10.6-12.3x slower | **2.29x FASTER** |
| conv2d | 4-6x slower | **1.27x FASTER** |
| BatchNorm2d f32 grad | 10x slower | 1.21x slower (parity) |
| `avg_pool1d` grad | 4-7x slower | 1.92x slower |

**Three of the five rows are wins, and neither remaining row is a 2x gap.** The
loss list as written described a codebase that no longer exists. The single
largest real vs-PyTorch gap on it is `avg_pool1d` at 1.92x.
