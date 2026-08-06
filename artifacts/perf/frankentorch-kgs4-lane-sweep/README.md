# kgs4 — op-work re-baseline of the gauntlet lanes that never had a live PyTorch arm

## Why this sweep exists

Five gauntlet lanes had never been compared against a live PyTorch arm under
`--features fair-alloc`. The re-baselined loss list
(`artifacts/perf/frankentorch-ug4ep/`) covered `avg_pool1d`, `batch_norm2d`,
`linear`, `conv2d` and GroupNorm, and every one came back a win or near-parity —
which made the campaign look finished. It was not finished; it was **incomplete**.

## Why op work only, not the whole step

`frankentorch-ujw3g` established that a whole-step gauntlet ratio is dominated by
each lane's per-iteration input rebuild: for `avg_pool1d`, 57% of the step was the
caller's 32 MiB `to_vec()` and only ~12% was the pooling forward, and the op work
was at ~1.18x while the whole-step ratio read 1.5–2.0x. A whole-step ratio mostly
measures buffer-copy cost, which is allocator-shaped and settled by
`frankentorch-1ji9l` option C.

So this harness (`crates/ft-api/examples/gauntlet_lane_sweep_h2h.rs`) times
**forward + backward with the leaf built outside the timed region on both sides**.
That is the number a kernel lever could actually move.

## Result

```
executing_elf_sha256=d09360645b37cc3ba81571ff5382cd3389d4209f20c0f8f102e7aaeac56dce2c
allocator=mimalloc (--features fair-alloc)
measurement=OP WORK ONLY (forward+backward; leaf built outside the timer on BOTH sides)
reps=15, PyTorch min-of-7 after 4 warmups, torch threads=8
```

| lane | FT (ms) | PT (ms) | standing | A/A gate | parity |
|---|---|---|---|---|---|
| **`max_pool3d`** `[2,32,16,32,32]` k2 s2 | 6.196 | 0.660 | **9.39x SLOWER** | **PASS** `[0.803,1.181]` | match |
| **`avg_pool2d`** `[8,64,64,64]` k2 s2 | 7.864 | 1.833 | **4.29x SLOWER** | **PASS** `[0.717,1.131]` | match |
| `max_pool1d` `[8,64,8192]` k2 s2 | 17.180 | 13.953 | 1.23x | **FAIL** `[0.843,0.986]` | match |
| `conv3d` `[2,32,8,16,16]` w`[32,32,3,3,3]` | 20.272 | 5.645 | 3.59x | **FAIL** `[0.770,0.992]` | match |

Every lane's gradient sum agrees with PyTorch's to 1e-6 relative, so all four are
comparing the same computation.

## What is quotable and what is not

**Quotable: `max_pool3d` 9.39x and `avg_pool2d` 4.29x.** Both passed their A/A null
gate (the same arm against itself, CI bracketing 1.0) and both match on parity.
`max_pool3d` is now the largest confirmed vs-PyTorch gap in the tree.

**NOT quotable: `max_pool1d` and `conv3d`.** Their A/A gates **FAILED** — the
null CIs are `[0.843,0.986]` and `[0.770,0.992]`, both excluding 1.0 on the high
side, meaning one arm was systematically faster than the other when both arms are
the identical code. That is a defect in the measurement, not a result, and it
makes those two ratios undecidable. They are recorded here only so nobody
re-derives them and believes them.

The likely cause is intra-iteration drift that the alternating assignment did not
fully cancel; the fix is to randomise arm order per iteration rather than
alternate, and to re-run on a quiet host. Until that is done, treat `conv3d`'s
3.59x as *unmeasured*, not as a loss.

## Do not compare these to the gauntlet's numbers

Op-work ratios and the gauntlet's whole-step ratios measure different things. The
gauntlet includes each lane's input rebuild; this does not. Neither is wrong, and
neither is comparable to the other.

## Reproducing

```
PYTORCH_PYTHON=<venv>/bin/python \
  cargo run --release -p ft-api --features fair-alloc --example gauntlet_lane_sweep_h2h
```

Must run locally — rch workers have no PyTorch, and the harness hard-fails rather
than reporting an FT-only number if the PyTorch arm did not run.
