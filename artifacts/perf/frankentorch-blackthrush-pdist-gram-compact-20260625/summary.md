# pdist f32 p=2 Gram-Buffer Compaction Reject

Agent: BlackThrush
Date: 2026-06-25
Worktree: `/data/projects/.scratch/frankentorch-blackthrush-boldverify-20260625T210520Z`
Target dir: `/data/projects/.rch-targets/frankentorch-cod-a`

## Lever

For no-grad contiguous f32 `tensor_pdist(x, p=2.0)`, shape `512x64`, reuse the
full `sgemm_bt` Gram matrix as the final condensed output buffer by writing the
strict upper triangle into the earlier slots and truncating. This avoids the
second `Vec` allocation and push loop while preserving the same Gram values and
output order.

## Result

- Baseline FT Criterion mean: `971.13 us`.
- Candidate FT Criterion mean: `1.0117 ms`.
- Internal speedup: `0.96x` (regression).
- PyTorch 2.12.1+cpu `torch.pdist` mean: `0.051994 ms`.
- Baseline ratio vs PyTorch: `18.68x slower`.
- Candidate ratio vs PyTorch: `19.46x slower`.

Decision: reverted source; commit only the negative evidence.

## Commands

Baseline:

```bash
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
  rch exec -- cargo bench -p ft-api --bench cdist_bench -- \
  pdist_f32_p2_mm/512x64 --warm-up-time 1 --measurement-time 3 \
  --sample-size 10 --noplot
```

Candidate:

```bash
AGENT_NAME=BlackThrush RCH_MIN_LOCAL_TIME_MS=99999999 \
  CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
  rch exec -- cargo bench -p ft-api --bench cdist_bench -- \
  pdist_f32_p2_mm/512x64 --warm-up-time 1 --measurement-time 3 \
  --sample-size 10 --noplot
```

Validation before revert:

```bash
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
  rch exec -- cargo test -p ft-api \
  pdist_p2_f32_fused_nograd_matches_composed_path --lib -- --nocapture
```

Conformance after revert:

```bash
AGENT_NAME=BlackThrush CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-a \
  cargo test -p ft-conformance -- --nocapture
```

Result: green. `ft-conformance` passed all lib, bin, integration, smoke, and
doc-test targets. Two `rch exec` attempts for this gate queued without starting,
so the final gate was run directly with the same warm target directory.
