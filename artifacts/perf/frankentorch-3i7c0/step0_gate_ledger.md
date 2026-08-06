# frankentorch-3i7c0 STEP 0 — is the large-buffer allocation churn real?

> **STATUS: CRITERION (a) ONLY. THIS IS NOT A GATE VERDICT, AND IT DOES NOT OPEN
> STEP 1.** The bead's gate has two criteria and only the first has been
> measured:
>
> - **(a) does large-buffer traffic persist across steps when inputs are reused?**
>   Measured. Reads **PASS** — the churn does not vanish. 3i7c0 therefore cannot
>   be rejected on the "it is only a benchmark artifact" ground.
> - **(b) does the allocator swap move the loop measurably?** **NOT RUN.** The
>   `--features fair-alloc` arm compiles clean but has not been executed.
>
> STEP 1 opens only when both pass. Nothing here authorises writing a pool.

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

```
executing_elf_sha256 37b39d68de74483c48a9a234d662ec24534fcd8e0badc25b6b0bc0c97d416507
allocator            system (default build)
large-block threshold 1048576 bytes | steps 24 (first 6 discarded)
```

| lane | step ms | steady-state large blocks/step |
|---|---|---|
| `pool_rebuild` | 31.379 | 4 alloc, **96.00 MiB** |
| `pool_reuse` | 17.873 | 3 alloc, **64.00 MiB** (step 0: 4 / 96.00 MiB — the one-time input build) |
| `mlp_reuse` | 64.040 | 15 alloc, **58.12 MiB** |

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
faults pages — was unaffected, which is why criterion (a) is still readable from
the table above.

**The table above is from the PRE-FIX binary** (`executing_elf_sha256`
`37b39d68…`), and the fixed source has not yet produced a run. An attempt to
re-run it did not do what it looked like it did: the rebuild was issued inside an
`rch exec -- sh -c "…"` wrapper, which suppressed rch's artifact sync, so the
local binary was never replaced and the "re-run" executed the same stale ELF —
the identical SHA and the still-zero free column are the tell. Re-issued as a
direct `rch exec -- cargo build`. Recorded because a self-A/B that silently
measures the same binary twice is exactly the failure mode this campaign's
ELF-SHA rule exists to catch, and here the rule caught it.

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
