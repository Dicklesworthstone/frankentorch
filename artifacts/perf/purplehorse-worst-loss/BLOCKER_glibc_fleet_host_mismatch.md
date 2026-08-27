# BLOCKER: the rch fleet builds against GLIBC_2.43; this host is 2.42 — no vs-torch measurement is possible

**Every h2h measurement in this campaign is currently unobtainable, and the cause is neither
contention nor a FrankenTorch defect. It is a toolchain version gap between the build fleet
and the measurement host.**

## The failure

```
$ RCH_REQUIRE_REMOTE=1 env -u CARGO_TARGET_DIR rch exec -- \
    cargo build -j2 --release --features fair-alloc \
    -p frankentorch-api --example bidiag_gate_sweep_h2h
$ ldd target/release/examples/bidiag_gate_sweep_h2h
  .../libm.so.6: version `GLIBC_2.43' not found (required by ...)
```

`cargo` exits **0**. The binary is produced, is the right size, and looks entirely healthy.
It cannot execute on this host.

```
host:    ldd (Ubuntu GLIBC 2.42-0ubuntu3.1) 2.42
ELF:     objdump -p → Version References: GLIBC_2.43   (libm)
```

## Why it blocks everything

The h2h harness must run **locally**: PyTorch exists only on this host
(`/data/tmp/torchvenv-2121`), and the whole methodology depends on driving the incumbent as
a co-process *inside the same invocation* — cross-run comparison is invalid here (the
incumbent has moved 1.94x between two runs of the same ELF).

But under the rch-only rule the harness can only be **built remotely**. Local builds are
barred: one previously consumed 119 GB of host disk.

Those two constraints are now mutually unsatisfiable.

## It is not contention, and not luck

* ~38 consecutive unloadable ELFs across many hours.
* `rch queue` showed **13 of 15 workers available, 0 busy** during several of them.
* Workers that produced loadable ELFs earlier the same day (`vmi1149989`, `vmi1152480`)
  now produce 2.43 binaries too — the fleet moved, it did not get busy.
* One retry burst *did* succeed on attempt 1 after 12 failures, so routing variance exists,
  but it has not recurred in ~38 attempts since.

## Distinct from frankentorch-c1ct1

`c1ct1` is worker-side **ENOSPC** surfacing as `rust-lld` SIGBUS under a 606-example link.
Different mechanism, different symptom, and its own bead scopes the fix to the RCH repo.
This one is a **glibc symbol-version gap** and produces a linkable, unrunnable ELF.

## Workarounds considered and rejected

| option | why rejected |
|---|---|
| build this one example locally | violates the hard rch-only rule; needs owner authorization (the rule targets 119 GB *workspace* builds, so one example target is arguably a different scale — but that is not my call) |
| `x86_64-unknown-linux-musl` static build | musl swaps the allocator. The harness is built `--features fair-alloc` (mimalloc) precisely because allocator behaviour moves these numbers. It would run, and its timings would not be comparable to any banked row — a measurement that looks valid and is not |
| drain the 2.43 workers | shared-state surgery while peers hold in-flight jobs |
| FT-vs-FT self-comparison | not a vs-incumbent measurement; explicitly excluded |

## What is stranded behind it

Five landed, correctness-gated levers with no vs-torch confirmation:

| change | FT-vs-FT | commit |
|---|---|---|
| `eigh` tridiagonal reduction parallelised (gated `l>=384`) | 1.40x lane @ n=1024 | `b091b458`, `ce9c3275` |
| `geqrf` trailing update via `dgemm_sub_into` | 1.082x | `d1180060` |
| `geqrf` `leaf` 8→2 | 1.145x / 1.175x | `526267af` |
| getrf `NB` 64→128 (three sizes) | 1.13–1.21x | `967a98e0` |
| blocked-QR leaf row-negation bug | correctness | `380699a2` |

## What unblocks it

Any one of: fleet/host glibc parity restored; authorization to build this single example
locally; or PyTorch installed on a worker so the harness can run where it is built.

## Detection recipe

`cargo` returning 0 is **not** a usable success signal here. The check that works:

```bash
B=target/release/examples/bidiag_gate_sweep_h2h
miss=$(ldd "$B" 2>&1 | grep -c "not found")
age=$(( $(date +%s) - $(stat -c %Y "$B") ))
```

Both are required. `ldd` alone passes on a **stale** binary — a 10:52 ELF once passed the
symbol check hours later and would have measured pre-change code. And note `find -newermt`
is unusable for the age check on this host: `find` is `bfs`, which rejects relative
timestamps and silently makes the condition unsatisfiable.
