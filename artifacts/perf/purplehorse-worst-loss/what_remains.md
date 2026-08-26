# What remains vs the live incumbent — an honest inventory

The worst loss is profiled to exhaustion at every level I can reach. This records
what is left, and is explicit about which parts do **not** meet the standard the rest
of this session held.

## SVD square forward — the worst op, and every lever on it is measured

**3.10–3.12x SLOWER than PyTorch 2.12.1+cpu at n=1024, CERTIFIED** (A/A null 0.991,
`elf_sha256=323a90cf…`, commit `7cf74314`). Confirmed worst by exclusion against all
58 board lanes (`73a050af`).

| lever | verdict | commit |
|---|---|---|
| expansion phase (`form_p`+`form_q`) | 33.5% of n=512; removing **all** of it still leaves 1.734x | `c4d611c4` |
| reduction parallel dispatch | **<5%** — below the instrument's own floor | `95a70cd8` |
| `n>=130` `form_p` blocking gate | measured **no-op** (1.378x vs 1.377x across it) | `7ebc0555` |
| `x86-64-v3` AVX2 | ~1.07x at n=512, nothing at n=256; flag proven present by disassembly | `636ce5bc` |
| row-dot vectorisation | already hand-written `wide::f64x4`; AVX2 halves packed ops 6→3 exactly as 256-vs-128-bit predicts | `feb83d5a` |
| FMA in the row-dot | **0 in both builds, by design** — contraction would change rounding and break bit-exactness | `feb83d5a` |

Nothing here is mis-implemented. The remaining gap is not recoverable by dispatch
choice, gate tuning, compiler flag, or instruction selection on the hot loop.

## The rest of dense linalg has NO in-invocation torch arm

This is the honest gap, and it is a gap in the *instruments*, not a set of unmeasured
losses I am declining to chase.

`linalg_gap_sweep` (already built, `ft-kernel-cpu`) times the eig family, qr and svd —
but **FT-internal only**. Its own header says the matrix is "identical … reproducible
in torch", i.e. torch is meant to be run *separately*. That is a **cross-run**
comparison, and this host has moved the incumbent 1.94x between two runs of the same
ELF. Every standing in this session was measured against a torch **co-process inside
the same invocation** precisely to avoid that, so I will not quote a torch ratio from
this harness.

What it *can* say, because relative costs inside one invocation are common-mode:

```
./target/release/examples/linalg_gap_sweep        threads=64
n=  256  eigvalsh   7.23  eigh  12.11  qr   5.72  svdvals  12.16  svd   18.96 ms
n=  512  eigvalsh  42.27  eigh  78.33  qr  38.87  svdvals  70.39  svd   95.98 ms
n= 1024  eigvalsh 333.73  eigh 549.19  qr 466.93  svdvals 619.47  svd  962.36 ms
```

Three things follow, all internal and none a torch ratio:

* **SVD is the heaviest op in our own linalg family** at every size — 95.98 ms at
  n=512 against eigh's 78.33 and qr's 38.87. Consistent with it being the worst
  standing rather than an artefact of where I happened to look.
* **`svdvals` is 73% of `svd` at n=512** (70.39 / 95.98) and 64% at n=1024. That
  independently reproduces the values-only arm's split measured at 8 threads
  (50.092 / 73.969 = 68%, `c4d611c4`) on a different thread count and a different
  harness.
* **64 threads is slower than 8 on this op**: svd n=512 reads 95.98 ms here at
  `threads=64` against 63–74 ms in the 8-thread runs. Consistent with the standing
  finding that the board's wide default penalises compute-bound lanes.

## Ops with a live-torch in-invocation harness, and therefore real standings

* **SVD family** — `bidiag_gate_sweep_h2h`. Certified above.
* **58 board lanes** — `gauntlet_lane_sweep_h2h`: conv2d (f32/f64, masked/summed/train),
  conv3d, max/avg pool 1d/2d/3d, group_norm, batch_norm, linear, attention. Swept
  `73a050af`; only six losses carry both nulls, worst `conv2d_f32_masked`.

## Ops with NO live-torch in-invocation harness

`eigh`, `eigvalsh`, `qr`, `cholesky`, `lu`, `solve`, `inv`, `lstsq`, `pinv`,
`matrix_exp`, `det`, `matrix_rank`, non-symmetric `eig`.

Their existing harnesses are either FT-internal sweeps (`linalg_gap_sweep`,
`qr_stage_profile_run`, `eigh_stage_profile_run`) or single-purpose probes. The prior
ledger records eigh at **1.36–3.59x FASTER** than torch and qr as peer-owned, and the
gap-find campaign closed with "all classes FT-faster or at ceiling" — but that ledger
has been wrong twice this session in both directions (item 257b's n=1024 was 25% too
harsh; a doc-comment-only 4.55x was exactly right), so those figures should be treated
as unverified rather than as standings.

**Building a co-process torch arm for the eig/qr family is the next instrument-level
piece of work**, and it is what would let anyone say whether SVD is the worst op in
the *tree* rather than the worst op *among ops that have a live-torch harness*. I have
been careful to claim only the latter.

## Still open, with the pieces already landed

`frankentorch-conv2d-mask-fusion-f64-only-iq1j1` — the f32 mask fusion, the one lever
left with real headroom (`conv2d_f32_masked` 2.62–2.78x focused). Kernel landed and
proven bit-exact (`0698f772`), create_graph guard test landed and passing
(`a3596f16`), wiring written and stashed pending its ~150-line create_graph arm.
