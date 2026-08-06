# frankentorch-ujw3g — avg_pool1d vs PyTorch, measured

## Why a new harness

The canonical lane (`pytorch_gauntlet_bench -- avg_pool1d`) **cannot produce a
vs-PyTorch ratio in this fleet**:

- rch workers have no PyTorch. Under `--features fair-alloc` on `vmi1149989` the
  FrankenTorch arms measured but the incumbent died:
  `gauntlet_avg_pool1d_grad/pytorch_2_12_cpu -> PyTorch avg_pool1d benchmark failed with status Some(1)`
  (torch import, `benches/pytorch_avg_pool1d_grad.py` line 4).
- rch does not sync bench binaries back either. `rch exec -- cargo build ... --bench
  pytorch_gauntlet_bench` reports `Finished`, but `find target -name 'pytorch_gauntlet*'`
  is empty locally — rch syncs `examples/`, not `deps/`. So "build remote, run
  local against the local venv" does not work for a bench.

Examples *do* sync, so `crates/ft-api/examples/avgpool1d_h2h.rs` runs locally with
the incumbent in-process. It uses the gauntlet's exact shape and adds an A/A null
gate, a landed-win anchor, and an in-process ELF SHA — none of which the criterion
lane has.

## Result

```
executing_elf_sha256=6bbcb383997093768060bb99669afdb1643b724c93d609e1d903dbff02c8dcaf
allocator=mimalloc (--features fair-alloc)
workload=avg_pool1d_grad_sum_loss [8,64,8192] f64 kernel=2 stride=2 reps=21
a_a_median_ratio=0.9709 ci95=[0.9293,1.0342] gate=PASS
  avg_pool1d f64 parity: gradient sum matches PyTorch (2.097152000000e6)
op            FT(ms)    PT(ms)   verdict
  cat_anchor     10.224   41.931   FT 4.10x FASTER
  avg_pool1d     33.601   17.142   FT 1.96x SLOWER
```

Parity is asserted **before** timing (FT gradient sum vs PyTorch's, 1e-9 relative);
a timing lane is worthless if the two sides compute different things. The A/A null
gate passes with its CI bracketing 1.0, and `cat_anchor` reads 4.10x against the
4.21x the same anchor gave in an independent `groupnorm_h2h` run earlier the same
session — so this is a sane measurement window, not a contended one.

**1.96x SLOWER**, which independently confirms the 1.92x recorded in
`artifacts/perf/frankentorch-ug4ep/`. The row survives re-measurement on a
different harness.

## Scope of this number — read before quoting it

This measures the **idiomatic compose**: `functional_avg_pool1d` -> `tensor_sum` ->
`tensor_backward`, which is what a user writes. It is **not** the same arm as the
gauntlet's `frankentorch_kgs4_134_fused_sum_loss`, and the two are not
interchangeable:

- PyTorch here runs at `torch.set_num_threads(8)`; the gauntlet defaults
  `FT_TORCH_THREADS=32`, which is why its PyTorch arm reads ~6 ms where this one
  reads 17 ms.
- The gauntlet's FT arm calls the **fused** scalar-loss path; this one does not.

So do not compare this 33.601 ms against the gauntlet's 7.4 ms. Both are valid
against their own in-invocation incumbent and neither is valid against the
other's.

## The lever's ceiling, measured before building it

The obvious lever is "route `tensor_sum(avg_pool1d(x))` to the existing fused
`functional_avg_pool1d_sum` automatically". Before writing it, the harness gained
a third arm that **calls the fused API directly** — no routing change can beat
that, so it is the lever's ceiling. Measured in the same invocation, interleaved
with the compose:

```
executing_elf_sha256=afead5e17d3414ec9e49a57021080a707da1da2f6352055d96a51fff33f4b6c1
allocator=mimalloc (--features fair-alloc)
a_a_median_ratio=1.0232 ci95=[0.9990,1.0484] gate=PASS
  fused vs compose: gradient sums are bit-identical
compose_ms=30.5753 fused_ms=26.6170 compose_over_fused=1.1487 ci95=[1.1006,1.1746]
op            FT(ms)    PT(ms)   verdict
  cat_anchor     11.237   44.837   FT 3.99x FASTER
  avg_pool1d     30.575   17.586   FT 1.74x SLOWER
```

**Ceiling = 1.15x**, CI lower bound 1.1006 so it clears 1.0 and is a real,
resolvable effect — but a modest one. Landing it perfectly would move this lane
from 1.74x slower to roughly 1.51x slower *on this run's pairing*. It does **not**
close the gap. (Do not confuse that projected 1.51x with the separately *measured*
1.51x of the later `bf81cc37…` run — they coincide by accident.)

### All three runs, so nobody quotes one as if it were precise

Every row below pairs FT with the PyTorch arm measured **in the same invocation**:

| ELF | FT (ms) | PT (ms) | ratio |
|---|---|---|---|
| `6bbcb383…` | 33.601 | 17.142 | 1.96x slower |
| `afead5e1…` | 30.575 | 17.586 | 1.74x slower |
| `bf81cc37…` | 31.211 | 20.647 | 1.51x slower |

The FrankenTorch arm is stable (30.6–33.6 ms); the **PyTorch arm carries almost
all the spread** (17.1–20.6 ms). So the honest statement of this row is
**"roughly 1.5x–2.0x slower"**, and any single-run figure quoted to three
significant figures is overstating what was measured. The phase split and the
fused-vs-compose ratio are both internal FT-vs-FT comparisons and are
correspondingly tighter.

This also corrects an inference I had drawn from the gauntlet's two FT arms
(`kgs4_122` 8.08 ms vs `kgs4_134_fused_sum_loss` 7.44 ms, CIs overlapping): those
were a single contended run on a worker and could not resolve the difference. A
proper interleaved measurement does resolve it, and puts it at 1.15x rather than
"not measurable" — so the fused path is genuinely better, just not by much.

**What this rules out.** The dense output-gradient buffer that the fused path
avoids is therefore *not* the dominant cost of this lane. With the fused arm still
1.51x off PyTorch, the remaining gap lives elsewhere — most likely in leaf
materialisation, which both arms pay and which the earlier phase-timing put at
~43% of the step. A future lever should target that, not the pooling backward.

## Where the time actually goes — this redirects the lever

Same harness, same invocation, under `--features fair-alloc`
(`executing_elf_sha256=bf81cc37…`, A/A null `1.0089 ci95=[0.9724,1.0518]` PASS):

```
phase_split materialise=18.232ms (57%) forward=3.176ms (10%) backward=10.747ms (33%) total=32.155ms
compose_ms=31.2110 fused_ms=28.6522 compose_over_fused=1.0893 ci95=[1.0414,1.1369]
cat_anchor  10.468   43.698   FT 4.17x FASTER
avg_pool1d  31.211   20.647   FT 1.51x SLOWER
```

| phase | ms | share |
|---|---|---|
| **materialise the input leaf** | **18.232** | **57%** |
| pooling forward + sum | 3.176 | 10% |
| backward + gradient read | 10.747 | 33% |

**FrankenTorch's input materialisation alone (18.232 ms) is 88% of PyTorch's
entire train step (20.647 ms).** Even a *free* pool forward and a *free* backward
would leave this lane at roughly parity at best, not ahead.

> **CORRECTION.** An earlier revision of this section paired this run's FT number
> with the **previous run's** PyTorch number (17.586 ms, from ELF `afead5e1…`) and
> claimed materialisation exceeded PyTorch's whole step. It does not — against its
> own same-run incumbent (20.647 ms) it is 88% of it. The corrected ratio for this
> run is **1.51x slower**, not 1.74x. Mixing arms across runs is precisely what
> this campaign's same-invocation rule exists to prevent, and the conclusion below
> is restated against the same-run pairing only.
>
> Worth noting in its own right: the PyTorch arm moved 17.586 -> 20.647 ms between
> two runs minutes apart, a 17% spread. That is why a ratio is only quotable
> against the incumbent measured beside it, and why the per-run pairing matters
> more than either number alone.

The pooling forward — the thing "avg_pool1d is slow" would lead you to optimize —
is **10%** of the step. The fused-routing lever targets the backward, so its
ceiling is bounded by 33%, and it measures 1.09x–1.15x across two independent
runs. Both are the smaller half.

### Is that comparison fair?

Yes, and it is worth being explicit because it is the crux. Both arms rebuild
their input every step: the PyTorch script does
`base.detach().clone().requires_grad_(True)`, this harness does `base.to_vec()`
into `tensor_variable`. Both copy the same 32 MiB. The difference is what that
copy *costs*: FrankenTorch's runs at roughly 1.8 GB/s, which is about 5x off
memcpy speed for a buffer that size, and is the signature of first-touch page
faults on freshly-obtained pages. PyTorch's caching allocator hands back a warm
block and never pays them.

Note this persists **under mimalloc**. `frankentorch-3i7c0` established that
swapping the allocator buys 1.95x on a rebuild-shaped lane, and this measurement
is already taking that win — 18.232 ms is the *post-mimalloc* number. So the
residual is not "we should use a better allocator"; that lever is already pulled.

## The lead this exposes

`functional_avg_pool1d_sum` already exists (`crates/ft-api/src/lib.rs:32466`,
backed by `avg_pool1d_sum_forward_f64` + `avg_pool1d_backward_scalar_f64`, which
avoids materialising the output-gradient buffer entirely). But **there is no
auto-shortcut from `tensor_sum(functional_avg_pool1d(x))` to it** — a user gets
the fast path only by knowing to call a differently-named API.

That is precisely the gap already closed for `group_norm`, where the fused
scalar-loss outputs are registered in a shortcut so `tensor_sum(group_norm(...))`
reaches the fused path automatically. The same treatment for `avg_pool1d` is the
obvious next lever, and it is a **routing** change rather than a new kernel: the
kernel it would route to is already written, already tested
(`functional_avg_pool1d_sum_matches_pool_sum_backward_bits`), and already
bit-exact against the compose.

**Not attempted, and now DEPRIORITISED on the phase split above.** It is a real
effect (1.09x–1.15x across two runs, CI lower bound clears 1.0, bit-exact) and it
is cheap, but it targets the backward — 33% of the step — while 57% sits in leaf
materialisation. Building it first would be attacking the smaller half of a gap
whose larger half is now measured and named.

Recommended order for whoever takes this next:

1. **Leaf materialisation (57%).** This is the one that matters and it is bigger
   than PyTorch's whole step. The mechanism to attack is first-touch page-fault
   cost on freshly-obtained large buffers — note the allocator lever is already
   spent (this is the post-mimalloc number), so the remaining move is on the
   FrankenTorch side of leaf creation, not the allocator.
2. **Fused routing (33% ceiling, 1.09x–1.15x measured).** Worth landing as a
   gap-close once the big term is addressed, since it makes the idiomatic path
   (`tensor_sum(avg_pool1d(x))`) as fast as the expert path without the user
   having to know a second API name. It is a routing change and the kernel it
   routes to is already written, already tested, and already bit-exact.
3. **The pooling forward is 10%.** Do not start here, whatever the bead title
   says.

## Reproducing

```
PYTORCH_PYTHON=<venv>/bin/python \
  cargo run --release -p ft-api --features fair-alloc --example avgpool1d_h2h
```

Must run locally (the workers have no torch). The harness hard-fails if the
PyTorch arm did not run, rather than silently reporting an FT-only number.
