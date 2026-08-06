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

**Not yet attempted, and deliberately not claimed.** Recorded as the lead so the
next agent starts from the routing question rather than re-profiling the kernel —
and note the prior root-cause finding that the pooling kernel itself is only
~1.4x off parity, so a kernel-aimed lever is chasing the smaller half of the gap.

## Reproducing

```
PYTORCH_PYTHON=<venv>/bin/python \
  cargo run --release -p ft-api --features fair-alloc --example avgpool1d_h2h
```

Must run locally (the workers have no torch). The harness hard-fails if the
PyTorch arm did not run, rather than silently reporting an FT-only number.
