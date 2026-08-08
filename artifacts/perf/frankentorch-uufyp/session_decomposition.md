# uufyp — `tape_overhead` split: it is the BACKWARD tape pass, ~34x its own kernel

`k1h8g` found 91–93% of avg_pool2d's session arm sits outside the kernels but could
not say *where*. Split into cumulative stages, the answer is one term.

## Measured

`crates/ft-api/examples/pool_kernel_vs_tape_probe.rs` (new decomposition section),
current HEAD, mimalloc, 15-rep medians, three runs, load ~6–11.
Lane: `avg_pool2d [8,64,64,64]` f64 — a 16 MiB input.

Cumulative lanes, each adding one stage; the table is the successive differences.

| stage | run 1 | run 2 | run 3 | vs its own kernel |
|---|---|---|---|---|
| `session_new` | 0.000 | 0.000 | 0.000 | — (free) |
| `leaf_build` | 0.945 | 0.695 | 0.877 | ≈ a bare 16 MiB clone |
| `forward` | 2.610 | 2.441 | 2.223 | raw_fwd 0.66–0.89 → ~1.5–1.7 ms over |
| `sum` | 0.380 | 1.283 | 1.040 | — |
| **`backward`** | **11.750** | **10.949** | **10.914** | **raw_bwd 0.32 → ~34x** |
| `grad_fetch_sum` | 1.139 | 0.956 | 2.030 | — |
| *(session total)* | 16.825 | 16.324 | 17.084 | |

## The finding

**`tensor_backward` costs ~11 ms while the backward kernel it dispatches costs
0.32 ms.** That single stage is ~65–70% of the entire session arm and roughly 34x the
work it exists to perform. Nothing else is close:

- `session_new` is **free** (0.000) — creating a session costs nothing, so the
  "fresh session per iteration" framing was not the problem.
- `leaf_build` is ~0.7–0.95 ms and is **just the 16 MiB input clone** — the session
  adds nothing measurable on top of `Vec::clone`.
- `forward` carries ~1.5–1.7 ms of tape cost over its kernel. Real, but a seventh of
  the backward term.

So of `k1h8g`'s 91–93%, the overwhelming majority is **one stage**, and it is the one
the DAC/autograd engine owns rather than anything in ft-kernel-cpu.

## What is NOT attributable here, and why I am saying so

The small stages are at or below this harness's noise, and one row proves it: the
bare-clone control came out at **164% / 128% / 234%** of `leaf_build`. A component
cannot exceed the stage containing it, so those three numbers are noise, not
measurement. `leaf_build`, `sum` and `grad_fetch_sum` should be read as "roughly a
millisecond each, not separable", **not** as the precise values in the table.

The `backward` term is the opposite case: 10.9–11.8 ms across three runs, ~10x any
plausible noise in the neighbouring stages, and consistent to within 8%. That one is
safe to act on.

## What this does and does not license

**Licensed:** the next lever on this lane belongs in `ft-autograd`'s backward pass,
not in `ft-kernel-cpu`. `k1h8g` should be re-scoped accordingly, and `87sz8`'s lane
is likely the same shape (it showed 62–69% outside the kernels).

**Not licensed:** "optimise `tensor_backward`". That is now the *unit* — the same
error one level down that `8obhh` made with "the dense write" and that this bead was
filed to avoid making with "the tape". The 11 ms is still unattributed *within* the
backward pass: candidates include gradient-slot allocation, DAC evidence recording,
node traversal, and the retention issue already tracked in
`project_gmuml_tape_retention` (the session tape reportedly never frees nodes). Split
*that* before choosing a lever.

**And whatever comes out of it must be A/B'd at the lane**, with a negative control,
per `8obhh`: a 4.24x isolated kernel win turned into a 1.13x lane loss because the
isolated framing missed the working set.

## Caveats

Single lane (`avg_pool2d`), single shape, one host. The absolute figures include the
probe's framing (fresh session, leaf clone and gradient sum inside the timed region),
so they are **not** comparable to the h2h's op-work numbers — see `k1h8g`'s artifact
for why those two must not be differenced. The *proportions* are what this file
claims. No PyTorch arm; nothing here is a vs-upstream claim.
