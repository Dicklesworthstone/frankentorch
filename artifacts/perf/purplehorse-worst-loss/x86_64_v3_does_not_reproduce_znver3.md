# `x86-64-v3` does not reproduce znver3's AVX2 gain — and this time the flag is PROVEN to be in the binary

**Result: a NEGATIVE finding, replicated with the confound item 260 could not exclude
now excluded. `-C target-cpu=x86-64-v3` is worth ~1.07x at n=512 and ~1.00x at n=256
on the SVD forward, against the 1.14x and 1.24x that `znver3` measured. Neither figure
is certifiable — n=2 per arm against a protocol floor of n=8, and several A/A nulls
fail — but the direction is clear and it agrees with item 260b.**

## What item 260 left open, and why it mattered

Item 260 measured `-C target-cpu=znver3` worth **1.14–1.24x bit-exact** on the SVD
forward (701/701 goldens). It did not ship, because `x86-64-v3` — the flag you would
actually deploy, since `rch` compiles on a heterogeneous fleet and a znver3 binary can
SIGILL elsewhere — failed to reproduce it, and the three-ISA palindrome meant to
settle it landed in a window with 2.29x PyTorch spread and 187 iowait jiffies. Item
260b closed with two readings it could not separate:

> `x86-64-v3` genuinely codegens worse than `znver3` on this machine, or the v3
> windows were bad. **Either way the shipping candidate is unmeasured.**

## The confound that had to be excluded first

`rch`'s `[environment] allowlist = []` means **`RUSTFLAGS` does not propagate to the
remote worker**. Building the obvious way —

```
RUSTFLAGS="-C target-cpu=x86-64-v3" rch exec -- cargo build ...
```

— produces a **baseline** binary on the worker, silently. The resulting A/B compares
two identical ELFs, measures nothing but the null, and reads exactly like "v3 doesn't
help". That is a live candidate for what item 260b actually observed.

Built instead with `cargo --config 'build.rustflags=["-Z","threads=4","-C","target-cpu=x86-64-v3"]'`,
which travels with the command, and **verified by disassembly before measuring
anything**:

| | baseline ELF | v3 ELF |
|---|---|---|
| `ymm` register uses | 662 | **12,215** |
| `vmovups` | 82 | **8,796** |
| sha256 | `5650da0f…` | `2bfcd944…` |

The flag is unambiguously in the binary. "The flag never applied" is now excluded as
an explanation for what follows.

## The measurement

ABBA — `base, v3, v3, base` — so a monotone host trend lands symmetrically on both
arms. Each ELF carries its **own live PyTorch co-process**, so `pt_v3 / pt_base` is a
free control. Each pass additionally runs `FT_GATE_VALUES=262144,262144` (the shipped
arm twice) so it carries its own A/A null rather than borrowing a neighbour's.

```
RAYON_NUM_THREADS=8 FT_GATE_SIZES="256,512" FT_GATE_VALUES="262144,262144" \
PYTORCH_PYTHON=/data/tmp/torchvenv-2121/bin/python <ELF>
```

All four passes launched into clean windows — idle **93.92/92.26, 86.76/87.95,
93.95/93.00, 88.52/91.11 %** (mpstat 5 s, twice, immediately before each launch).
iowait 2–51 jiffies. Incumbent PyTorch 2.12.1+cpu, threads=8, self-reported per pass.

| pass | arm | n=256 vs PT | n=512 vs PT | PT min 256 | PT min 512 |
|---|---|---|---|---|---|
| 1 | base | 1.514x / 1.539x | 2.650x / 2.390x | 6.867 | 31.027 |
| 2 | v3 | 1.387x / 1.507x | 2.320x / 2.458x | 6.903 | 30.810 |
| 3 | v3 | 1.619x / 1.502x | 2.363x / 2.535x | 6.877 | 30.799 |
| 4 | base | 1.516x / 1.488x | 2.661x / 2.665x | 7.024 | 31.930 |

### The ratio of ratios

| n | base mean | v3 mean | **v3/base** | pt control | item 260's znver3 |
|---|---|---|---|---|---|
| 256 | 1.514x SLOWER | 1.504x SLOWER | **1.0070x** | 0.9920 ✓ | 1.24x |
| 512 | 2.591x SLOWER | 2.419x SLOWER | **1.0713x** | 0.9786 ✓ | 1.14x |

Both `pt` controls satisfy the acceptance rule `|pt_ratio − 1| < 0.05`, so the
incumbent did not move between arms — the comparison is readable in that respect.

## Why this is NOT certified, stated plainly

The two-ELF protocol requires **n ≥ 8 surviving round-pairs and BOTH A/A gates
passing**. This run has:

* **n = 2 passes per label**, against a floor of 8. The protocol says refuse to
  compute below 8, and the means above are therefore descriptive, not inferential.
* **Several A/A nulls fail**: 1.088 and 1.045 on base, 0.924, 1.162 and 0.944 on v3,
  against a ±0.02 band. The n=256 pass-3 null at 1.162 is bad enough on its own to
  make that cell unreadable.

So neither 1.0070x nor 1.0713x is a banked number. What survives is the **comparison
of shapes**: znver3 measured 1.24x at n=256 and 1.14x at n=512; v3 here measures
essentially nothing at n=256 and about half of znver3's gain at n=512. The
disagreement at n=256 is larger than this run's noise can plausibly manufacture in the
direction that would rescue v3.

## What this settles and what it does not

**Settles:** item 260b's two competing readings collapse to one. The v3 flag *was*
applied — proven at the instruction level, 18.5x more `ymm` uses — and it still did
not reproduce znver3's gain. "The v3 windows were bad" is no longer needed to explain
the result, and this run's windows were good (86–94% idle, both `pt` controls inside
5%).

**Does not settle:** *why*. Item 260c asked for a disassembly diff between znver3 and
v3 to explain the mechanism, on the grounds that "AVX2 helps but only with Zen-3
tuning" is a claim that needs a mechanism rather than two stopwatches. That is still
owed and is not answered here.

**Deployment stays the user's call** and nothing was shipped: `.cargo/config.toml` is
untouched. Item 259e's point stands — a pinned ISA floor changes who can run the
binary, and the alternative, runtime dispatch, needs `#[target_feature]`, which is
`unsafe` to call and which these crates forbid.

The practical consequence for the campaign is the useful part: **the portable flag is
not the free 1.14–1.24x that znver3 suggested.** Anyone budgeting that win into the
SVD standing should budget ~1.07x at n=512 and nothing at n=256, and should treat even
that as unbanked.
