# frankentorch-3i7c0 STEP 0 — is the large-buffer allocation churn real?

> **STATUS: SPLIT VERDICT — (a) PASSES, (b) FAILS. The gate requires both, so the
> reading is REJECT.**
>
> - **(a) does large-buffer traffic persist across steps when inputs are reused?**
>   **PASS.** It does — 64 MiB/step in the pooling lane and 58 MiB/step in the
>   parameterised training lane, with allocations and frees exactly balanced.
> - **(b) does the allocator swap move the loop measurably?** **FAIL on every
>   realistic lane.** mimalloc moves the harness-shaped rebuild lane **1.95x**
>   (disjoint ranges over 3 interleaved reps) and moves neither reuse lane at all
>   (1.03x and 0.97x, ranges overlapping).
>
> The two together say something sharper than either alone: **the churn is real,
> but it is not costing anything.** A pool would be built to remove traffic that
> the measurement says is already effectively free in steady state. STEP 1 does
> not open.

## Why this gate exists

Every number motivating in-session buffer pooling came from
`pytorch_gauntlet_bench`, whose lanes rebuild their input tensor inside `b.iter`
on every iteration. Real training does not do that: it allocates parameters and
an input batch once and reuses them across steps. The operator therefore gated
the whole bead on one question — **does the churn survive when inputs are
reused?** A FAIL here is a success: it saves the entire pooling implementation.

## Method

`crates/ft-api/examples/training_loop_alloc_profile.rs`. The gate is answered by
**direct measurement of allocation volume**, not inferred from a timing delta: a
counting `GlobalAlloc` wraps whichever allocator the build selected and tallies
alloc/free count and bytes for blocks `>= 1 MiB` — the mmap-served class the
whole hypothesis is about — sampled per step.

Three f64 lanes in one process:

| lane | shape |
|---|---|
| `pool_rebuild` | `avg_pool1d [8,64,8192]`, input `Vec` and session rebuilt every step — the exact harness shape the pooling claim came from |
| `pool_reuse` | the same op, input leaf built **once**, each step freeing only its own graph generation via `truncate_autograd_graph(boundary)` |
| `mlp_reuse` | parameterised step: weight `[1024,1024]` + bias + batch `[1024,1024]` allocated once, `linear -> relu -> sum -> backward`, gradients applied **in place** by an SGD update, graph generation freed each step |

Two details that decide how the numbers should be read:

- `zero_grads_tensor` fills the persistent gradient buffer in place
  (`Arc::make_mut(g).fill(0.0)`); it does **not** drop it. The accumulation
  buffer is therefore allocated once and correctly does not count as churn, so
  the reuse lanes measure only genuinely transient intermediates.
- the two `avg_pool1d` lanes are asserted to produce the **bit-identical**
  gradient checksum, so the comparison is between two harness shapes of the same
  work, not between different work.

The mimalloc arm is a second build of the same file (`--features fair-alloc`),
because a global allocator is a compile-time, process-global choice. Both builds
print their allocator name and executing-ELF SHA-256, and both run on the same
local host.

## Cross-check performed first: the GroupNorm row, independently replicated

Before trusting any allocator claim, the loss-list re-baseline
(`artifacts/perf/frankentorch-ug4ep/loss_list_rebaseline_ledger.md`) was spot-checked
on a fresh binary in a separate session, since that ledger is what retired most of
the standing perf priorities:

```
$ PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
    ./target/release/examples/groupnorm_h2h
executing_elf_sha256=2ab0c47174cb48ac21f414c6b3ee028d1738b1e139701f3aa7ae36e2773cd583
workload=group_norm_f32_no_affine [16,256,64,64] groups=32 reps=31
a_a_median_ratio=1.0226 ci95=[0.9541,1.0842] gate=PASS
  group_norm f32 parity: 8 PyTorch probes within tolerance
  cat_anchor     10.980   46.199   FT 4.21x FASTER
  group_norm      3.480   10.350   FT 2.97x FASTER
```

A/A null gate PASS (CI contains 1.0), the landed-win `cat_anchor` reads its
expected value, parity holds against the live PyTorch arm in the same
invocation. **2.97x FASTER**, against the ledger's 2.94x — the row listed as
"19x slower" replicates as a win on an independent run.

## Result — arm 1 of 2, system allocator

Measured on the **fixed** binary (the earlier pre-fix table is preserved below for
provenance).

```
executing_elf_sha256 2df11ebbd376837ba2d6250625536478fca31291fe1e29c0e74c74aa79f0c838
allocator            system (default build)
large-block threshold 1048576 bytes | steps 24 (first 6 discarded)
```

| lane | step ms | steady-state large blocks/step |
|---|---|---|
| `pool_rebuild` | 32.238 | 4 alloc, **96.00 MiB** — frees not in window, see caveat |
| `pool_reuse` | 22.489 | 3 alloc **64.00 MiB**, 3 free **64.00 MiB** |
| `mlp_reuse` | 64.430 | 15 alloc **58.12 MiB**, 15 free **58.12 MiB** |

With the generation-free now inside the measured window, the two reuse lanes show
**allocations and frees exactly balanced every step** — 64.00 MiB out and 64.00 MiB
back, 58.12 MiB out and 58.12 MiB back. That balance is the definition of churn:
in steady state each step takes the same large buffers from the allocator and
hands them straight back. It is a stronger reading of criterion (a) than the
alloc column alone, because it rules out the alternative explanation that the
allocations are cumulative growth rather than turnover.

Caveat, stated because the two lanes are not symmetric: `pool_rebuild` still
reports zero frees. Its session is a local that drops when the step function
returns, which is after the closing snapshot, so its frees fall outside the
window. Its **alloc** column is correct, and the gate reads on the reuse lanes,
so this does not affect the verdict — but do not read `pool_rebuild`'s free
column as "it never frees".

## Result — arm 2 of 2, mimalloc, and the criterion (b) verdict

```
executing_elf_sha256 0fb3fff532cb8430c61ce86077e6792d4326e7e13fabfbcccb2833d650d2313e
allocator            mimalloc (--features fair-alloc)
```

Same host, same shapes, same step count. The two arms are distinct binaries with
distinct self-reported ELF SHAs and distinct self-reported allocator names, so
neither can be silently measured against itself.

Both binaries were preserved and run **interleaved, 3 reps each, alternating
arms on the same host in one sitting** (`system, mimalloc, system, mimalloc, …`),
so neither arm owns a warm or a cold slot. Medians below, with the full observed
range, because the range is what makes the verdict readable:

| lane | system (`a0df1848…`) | mimalloc (`0fb3fff5…`) | ratio | ranges |
|---|---|---|---|---|
| `pool_rebuild` (harness-shaped) | **32.466 ms** | **16.647 ms** | **1.95x faster** | `[31.955, 32.644]` vs `[16.293, 17.033]` — **disjoint** |
| `pool_reuse` (realistic) | **22.130 ms** | **21.393 ms** | 1.03x | `[21.416, 22.291]` vs `[20.866, 21.544]` — **overlapping** |
| `mlp_reuse` (realistic) | **63.640 ms** | **65.398 ms** | 0.97x (mimalloc slower) | `[63.504, 65.145]` vs `[65.100, 65.400]` — **overlapping** |

Large-block traffic is byte-identical on both arms in every lane (4/96.00 MiB,
3/64.00 MiB, 15/58.12 MiB), which is the control: the allocator swap changed how
the same allocations are served, not how many there are.

The separation on `pool_rebuild` is not a judgement call — the two arms' ranges do
not touch, and the gap between them is twenty times the width of either range. The
two reuse lanes' ranges **overlap**, so their 3% and -3% are not resolvable
differences at all; on `mlp_reuse` mimalloc is if anything marginally slower.

**Criterion (b) fails on every realistic lane.** The allocator swap is worth 1.87x
on the lane that rebuilds its input every iteration — the gauntlet's shape, and
the shape every number motivating this bead came from — and is worth nothing at
all on the two lanes that reuse their inputs the way training does.

### What the two criteria say together

They are not in tension; read jointly they are sharper than either alone.

The churn is **real** — the reuse lanes hand back 64 MiB and 58 MiB of large
blocks every step, allocations and frees exactly balanced. Criterion (a) was
right to pass. But that churn is **not costing anything**: swapping in an
allocator specifically designed to make repeated large-block traffic cheap
changes those lanes by 3% and -2%, which is noise.

So the original hypothesis was half right in a way that matters. The 2.79x and
3.96x figures that motivated pooling were real measurements of a real effect —
but the effect belongs to the harness's **per-iteration input rebuild**, not to
the steady-state churn of a training step. Reusing the input removes the
allocator's leverage entirely while leaving the churn volume untouched. That is
the cleanest possible demonstration that volume of allocation traffic is the
wrong proxy for cost.

The most likely mechanism, stated as a hypothesis and not as a measured claim:
glibc's `malloc` adapts its `mmap` threshold upward when it observes repeated
large frees, so a steady-state loop that recycles the same handful of buffer
sizes stops being served by `mmap`/`munmap` and starts being served from the
heap — i.e. glibc converges on caching behaviour by itself. Whatever the
mechanism, the decision does not depend on it: the measurement is that there is
no headroom left for a pool to recover.

### Superseded: the pre-fix run

Kept so the correction is auditable. Same lanes, binary
`37b39d68de74483c…`, whose free column was structurally zero because the
generation-free ran outside the window: `pool_rebuild` 31.379 ms / 4 alloc /
96.00 MiB; `pool_reuse` 17.873 ms / 3 alloc / 64.00 MiB; `mlp_reuse` 64.040 ms /
15 alloc / 58.12 MiB. The alloc columns agree exactly with the fixed run, which
is the evidence that the defect touched only the free accounting.

**The churn does not vanish when inputs are reused.** Rebuilding the input costs
96 MiB of `>= 1 MiB` blocks per step; reusing it still costs 64 MiB per step. The
parameterised training step — weights, bias and batch all allocated once,
gradients applied in place — still costs 58 MiB per step across 15 blocks.

The harness rebuild therefore accounts for exactly **one of the four** large
blocks in the pool lane (the 32 MiB input clone, a third of that lane's traffic).
The other 64 MiB is genuinely per-step: transient forward and backward
intermediates that any training loop allocates and frees every iteration.

That is the opposite of the outcome this gate was written to catch. On the
bead's criterion (a) — "large-buffer alloc/free traffic persists across steps" —
this reads **PASS**, and 3i7c0 cannot be rejected on the "it is only a benchmark
artifact" ground.

**This is not yet a verdict.** The gate has two criteria and the second — does
the allocator swap move the loop measurably — needs the `--features fair-alloc`
arm. Criterion (a) alone does not open STEP 1.

## A defect found in this probe before publishing it

The first version called `truncate_autograd_graph` **outside** the measured
window, so every free landed in the gap between snapshots and the free column
read `0.00 MiB` on all three lanes. Freeing a step's graph generation is part of
the step, and it now runs inside the window.

The alloc column — which is what the gate turns on, because allocation is what
faults pages — was unaffected, and the fixed run's alloc figures match the pre-fix
run's exactly, which confirms the defect touched only free accounting.

**Getting the fixed binary to actually run took three attempts, and the first two
silently ran the old one.** Recorded in full because a self-A/B that measures the
same binary twice is precisely the failure this campaign's ELF-SHA rule exists to
catch, and here it caught it twice:

1. The rebuild was issued inside `rch exec -- sh -c "…"`. That compiles on the
   worker and reports success, but **suppresses rch's artifact sync**, so the
   local binary was never replaced. The "re-run" executed the same ELF —
   identical `executing_elf_sha256` and a still-zero free column were the tell.
2. Re-issued as a bare `rch exec -- cargo build`, it failed with
   `error: couldn't find file crates/ft-api/examples/training_loop_alloc_profile.rs`
   on two different workers, for a path that exists locally, is git-tracked and is
   not ignored — a stale worker-side sync manifest from when the file was still
   untracked.
3. `rch sync --worker <id> --force` (the tool's own remedy for a stale worker
   cache, 14 cache entries removed) cleared it; the next build produced a new SHA
   `2df11ebb…` and the free column populated.

Operational rule worth carrying: when you need the built artifact back on the
local box, issue a **bare** `rch exec -- cargo build` — no `sh -c`, no pipes — and
compare the in-process ELF SHA across arms before believing any ratio.

## A blocker cleared on the way

Verifying this probe under `clippy -D warnings` fails before it ever reaches
`ft-api`, on three pre-existing `clone_on_copy` errors in
`crates/ft-autograd/src/lib.rs` (10571, 10623, 10676). Confirmed present on
`HEAD`, so pre-existing debt rather than peer WIP. This is the same class as the
already-closed `frankentorch-1zrvy`, which was the same failure in
`ft-kernel-cpu`.

Filed and fixed as `frankentorch-e1f8z`. All three sites are inside
`fn permute_slice<T: Copy + Send + Sync>`, so `.clone()` on a `Copy` `T` is
exactly a copy and removing it is a bit-identical no-op — and it matches those
sites' own comment, which already argues for "a direct element move" over
clone-based copying.

## VERDICT: REJECT

The gate required both criteria. (a) passes, (b) fails on every realistic lane,
so **`frankentorch-3i7c0` is rejected and STEP 1 does not open.** No pool is
built.

Restating the bead's own decision rule against what was measured:

> PASS (churn is real): large-buffer alloc/free traffic persists across steps
> **and** the allocator swap moves the loop measurably. Proceed to STEP 1.

The traffic persists. The swap does not move the loop. The conjunction fails.

**This outcome is the point of the gate, not a disappointment.** It saves the
entire pooling implementation — which the bead itself flagged as carrying the
project's highest-risk failure mode, a pooled buffer aliasing a tensor still
reachable from the tape, against the non-regression rule that autograd
correctness outranks kernel speed. Buying that risk for a measured 0% is a bad
trade, and now it is a measured bad trade rather than an intuition.

### What replaces it

Nothing. The allocator question is closed by `frankentorch-1ji9l`'s option C
(landed, `7f57fce3`): `fair-alloc` stays opt-in for FrankenTorch's own
measurement binaries, the library never sets a global allocator, and the gauntlet
prints which allocator produced its numbers. That combination already handles the
one place the effect is real — the bench's own per-iteration input rebuild.

### Retry predicate

Re-open **only** if a real user workload — not a benchmark — is shown to
repeatedly allocate large tensors in steady state **and** to be measurably moved
by an allocator swap. Both halves are required: this bead's whole finding is that
the first half alone is satisfied by workloads where the second half is not.

A concrete trigger that would qualify: a training or inference loop whose
allocation profile resembles `pool_rebuild` rather than `pool_reuse` — i.e. one
that genuinely cannot reuse its input buffers across steps — showing a
`fair-alloc` delta with non-overlapping ranges over interleaved reps.

### Reproducing this

```
cargo build --release -p ft-api --example training_loop_alloc_profile
cargo build --release -p ft-api --features fair-alloc --example training_loop_alloc_profile
```

Preserve each binary before building the other (they share an output path), then
run them alternately. Each prints its own allocator name and ELF SHA-256 from
inside the process, so a stale-binary mix-up is visible in the output rather than
silent. `STEPS=<n>` overrides the step count.
