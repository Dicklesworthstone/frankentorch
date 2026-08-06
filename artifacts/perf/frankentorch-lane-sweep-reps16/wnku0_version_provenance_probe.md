# `frankentorch-wnku0` — the named probe, and what it printed

Follow-through evidence for the version-provenance enforcement shipped in
`afb8b185`. Kept beside the re-bank because it is the same harness, and because
one of these runs is the contention evidence cited in the README's range caveat.

## The probe the bead required

> a harness run whose output carries the torch version self-reported by the
> Python child in the same invocation, plus a check that quoting a ratio without
> it is impossible from the printed row alone.

## What it printed — shipped ELF, canonical oracle

`executing_elf_sha256=934f2538e733cda6cda6493115fc13ce3bf6d0411502c4be5d7fa08e0ea81a3f`

```
executing_elf_sha256=934f2538e733cda6cda6493115fc13ce3bf6d0411502c4be5d7fa08e0ea81a3f
incumbent=PyTorch 2.12.1+cpu (self-reported by the arm, same invocation), threads=8
incumbent_rule=a delta whose incumbent arm moved (version, build, or measured time) is NOT a win
allocator=mimalloc (--features fair-alloc)
measurement=OP WORK ONLY (forward+backward; leaf built outside the timer on BOTH sides)
reps=16, PyTorch min-of-7 after 4 warmups, torch threads=8
```

Re-run against the other interpreter, same ELF, the line changes:

```
incumbent=PyTorch 2.13.0+cpu (self-reported by the arm, same invocation), threads=8
```

So the field tracks the actual arm rather than printing a constant — which is
the only version of this check worth having.

## Why it is enforcement and not decoration

`ft_api::harness_provenance::require_reported_version` returns
`Err(MissingIncumbentVersion)` when the arm did not self-report, and all three
live-torch harnesses take it with `?`. A harness that loses its Python probe
**exits with an error instead of printing ratios**, naming the marker, the rule,
and the fix. There is no "unknown" fallback, because an "unknown" row is exactly
the row someone would quote.

Unit-tested at `cargo test -p ft-api --lib harness_provenance`, 10/10, including
four negative cases a naive implementation passes on the happy path and fails
here: an absent marker is `None` rather than a blank version, an empty marker
payload is `None`, `require_` errors rather than defaulting, and the Python
probe must emit the exact marker the Rust parser reads.

## The interim ELF, and why it is recorded

An earlier build of the same change, `f078f91d…`, produced the same provenance
block against torch 2.12.1. It is named here only so the two digests in this
session's logs are accounted for; `934f2538…` is the shipped one. A first draft
used `panic!` for the enforcement, which UBS correctly flagged as
panic-in-library-code (5 critical). It was reworked into an error type through
the harnesses' existing `Result` — UBS then reported 0 critical, "OK No panic!
macros". That was a fix, not a suppression, and the enforcement is unchanged.

## The run that landed outside the banked range

The `934f2538…` probe against torch 2.12.1 ran on the quietest window of the
session (load ~6, no multi-thousand-percent neighbour) and read:

| lane | this run | banked range | in range? |
|---|---|---|---|
| `max_pool1d` | 1.19x | 1.14–3.18x | yes (low edge) |
| `avg_pool2d` | **2.79x** | 3.09–8.07x | **NO — below** |
| `max_pool3d` | 6.12x | 5.85–8.53x | yes |
| `conv3d` | 3.52x | 3.19–4.42x | yes |

**This does not refute the bank.** One run is not a measurement — that is the
rule this artifact exists to enforce, and it applies to runs that flatter us as
much as to runs that do not.

**What it does establish** is that the banked ranges are contention-inflated:
the one lane that broke out did so *downward*, on the quietest host of the
session, in the direction a contention-widened range predicts. `avg_pool2d`'s
true floor is below 3.09x. See the README's range caveat, which this run is the
evidence for.

Recorded rather than omitted, because an out-of-range observation that is
quietly dropped is how a range stops meaning anything.
