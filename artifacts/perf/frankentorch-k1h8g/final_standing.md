# k1h8g — final standing: no kernel lever exists, and the remainder is diffuse

This bead was opened as "avg_pool2d op work ~4-6x slower than PyTorch". After a chain
of measurements it closes as **not actionable as a kernel bead**, with the residual
gap characterised and bounded rather than fixed.

## Where the lane actually stands

| | value | source |
|---|---|---|
| vs PyTorch (op work) | **2.20x / 2.33x slower** | h2h, torch 2.12.1, A/A PASS, `87sz8` artifacts |
| bead title claims | 4–6x | stale since before this chain |
| FT's two kernels (fwd+bwd) | **~1.3 ms** | `pool_kernel_vs_tape_probe` raw lanes |
| PyTorch's **whole op** | 1.70 ms / 2.71 ms | same h2h runs |

**FrankenTorch's avg_pool2d kernels are already faster than PyTorch's entire op**,
forward and backward included. Make both kernels infinitely fast and the lane barely
moves.

## Why no lever was found, in order

1. **`zoqws`** — the "contended first-touch faults" model for the dense gradient write
   was refuted in situ; its landed lever was a 1.118x regression and was reverted.
2. **`un3os`** — the real kernel-level effect (sparse scatter into an L3-resident
   buffer, serial beats 64 threads) was found and shipped at 1.4–2.2x per kernel…
3. **`8obhh`** — …and then **reverted**, because at the lane it was a 1.13x *loss*
   (pooled n=31, CI [1.042,1.198], 31/31 rounds, clean conv3d negative control). A
   buffer-size gate cannot see the autograd tape's working set.
4. **`o5t00`** — the same gate on avg_pool's own backward was measured as an active
   ~1.27x loss, so the lever was never available to this lane anyway.
5. **`uufyp`** — the session cost is ~55–60% one stage, `tensor_backward`, at ~18x its
   own kernel (after correcting a 2x allocator-conditioning inflation).
6. **`27aci`** — all five named sub-mechanisms inside that stage were eliminated:
   pad/unpad (no pad node at padding 0), full-size root init (root is scalar),
   tape retention (flat over 8 cycles), create_graph (default is off), and the −0.0
   canonicalization pass is at most ~5–10%. **The cost is diffuse.**

## What closing this means, and what it does not

**Does not mean the lane is fast.** It is 2.2–2.3x slower than PyTorch and that number
stands unchallenged in the ledger.

**Does mean there is no kernel work left worth doing here**, and that the remaining
cost is engine overhead spread thin enough that no stage-granularity probe can resolve
it further. Advancing it needs a different instrument — instruction-granularity
profiling (perf / flamegraph), which nothing in this tree currently does — or a
decision to ledger the bound.

## For whoever picks the lane up

- Do **not** optimise `avg_pool2d_forward_f64` or `avg_pool2d_backward_f64`. They are
  not the constraint and `o5t00` already measured one gate on the backward as a loss.
- Do **not** add a sixth speculative sub-mechanism to `27aci`'s list. Five were tested;
  bring a profiler.
- The one concrete, untested, arguably-removable item left is the full-size **−0.0
  canonicalization pass** in `accumulate_tensor_gradient_owned`. It exists only to make
  the owned path bit-match the borrowed path, so removing it is a **parity** decision
  before it is a perf one — and autograd correctness outranks kernel speed.
- The harness carries `condition_allocator()` and an A/A null lane now. Reuse them;
  the reason they exist is that two byte-identical lanes once read 83% apart, and a
  later probe measured its own lane ordering as a 6.5x effect.

The bead title's "4-6x" is left as written rather than edited — it is a historical
record of what was measured when it was filed, and the current standing lives here.
