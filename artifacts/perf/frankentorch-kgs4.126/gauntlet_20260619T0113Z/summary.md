# frankentorch-kgs4.126 Gauntlet Summary

Verdict: rejected and reverted.

## Workload

- Criterion group: `gauntlet_max_pool1d_grad`
- Shape: `[8,64,8192]`
- Operation: f64 leaf tensor, `functional_max_pool1d(kernel=2,stride=2)`,
  scalar `sum`, backward.
- Reference: PyTorch `2.12.1+cpu`
- Host: `thinkstation1`

## Measurements

| Revision | FrankenTorch median | PyTorch median | Ratio vs PyTorch |
|---|---:|---:|---:|
| Candidate `ae4ace3b` | `184.41 ms` | `14.984 ms` | `12.31x` slower |
| Parent `eda26661` | `178.47 ms` | `16.199 ms` | `11.02x` slower |

Candidate vs parent median ratio: `1.033x` slower. Criterion reported no
significant improvement (`p=0.12`, no performance change).

## Commands

```bash
PYTORCH_PYTHON=/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python \
CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
cargo bench -p ft-api --bench pytorch_gauntlet_bench -- max_pool1d --noplot
```

The parent run used a detached worktree at `eda26661` with the same benchmark
harness applied.

## Validation

- `rch exec -- cargo check -p ft-api --bench pytorch_gauntlet_bench`: passed on
  `ovh-a` for the final harness.
- `rch exec -- cargo test -p ft-kernel-cpu max_pool1d_direct_matches_2d_h1_first_tie_forward_backward_bit_exact`:
  passed on `ovh-a`.
- `rustfmt --edition 2024 --check crates/ft-api/benches/pytorch_gauntlet_bench.rs`:
  passed.
