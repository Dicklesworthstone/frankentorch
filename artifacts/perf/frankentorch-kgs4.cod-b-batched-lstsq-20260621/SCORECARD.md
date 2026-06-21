# Batched lstsq QR proof bundle

Date: 2026-06-21
Agent: cod-b / IvoryDeer
Lane: `frankentorch-kgs4`

## Lever

Add `lstsq_qr_batched_contiguous_f64` and route no-grad f64
`tensor_linalg_lstsq(A, B)` for exact-batch `A[..., m, n]`, `B[..., m, rhs]`
through the existing 2-D QR least-squares kernel when `m >= n`.

The route is deliberately conservative: every plane must satisfy the QR
full-rank overdetermined contract. If any plane declines, the public API falls
through to the existing SVD-based batched `lstsq` fallback already present on
`origin/main`.

## Head-to-head

FrankenTorch command:

```text
AGENT_NAME=IvoryDeer CARGO_TARGET_DIR=/data/projects/.rch-targets/frankentorch-cod-b \
  rch exec -- cargo run --release -p ft-api --example batched_lstsq_h2h
```

FrankenTorch ran on RCH `ovh-a`; RCH workers do not have `torch`, so the PyTorch
CPU comparator ran locally with
`/data/projects/.venvs/frankentorch-pytorch-cpu/bin/python`, torch `2.12.1+cpu`,
8 intra-op and 8 inter-op threads.

| Shape | FT ms | PyTorch ms | Ratio | Checksum |
|---|---:|---:|---:|---|
| `B=100000 m=8 n=4 rhs=2` | `11.469` | `163.709` | `14.27x faster` | match, `-1.400442176e-1` |
| `B=20000 m=16 n=8 rhs=2` | `11.398` | `54.562` | `4.79x faster` | match, `6.660585681e-1` |
| `B=8000 m=32 n=16 rhs=2` | `28.766` | `52.378` | `1.82x faster` | match, `4.776251654e-1` |

Score: `3W / 0L / 0N` vs PyTorch for this pass.

## Verification

Completed:

- `rch exec -- cargo test -p ft-kernel-cpu lstsq_qr_batched_matches_looping_2d_and_defers --lib -- --nocapture`
- `rch exec -- cargo test -p ft-api tensor_linalg_lstsq_batched_qr_matches_looping_2d --lib -- --nocapture`
- `rch exec -- cargo run --release -p ft-api --example batched_lstsq_h2h`
- local PyTorch sidecar with matching checksum rows
- `rch exec -- cargo test -p ft-conformance --profile release`
- `git diff --check` over the source, docs, example, and artifact bundle
- `ubs crates/ft-api/examples/batched_lstsq_h2h.rs docs/NEGATIVE_EVIDENCE.md docs/RELEASE_READINESS_SCORECARD.md artifacts/perf/frankentorch-kgs4.cod-b-batched-lstsq-20260621/...`

The focused tests were rerun after the final borrow-cleanup edits. Full-source
UBS timed out after 240s, so the pre-commit UBS proof is scoped to the new
example/docs/artifact surface (`0` critical, `0` warnings). A scoped clippy run
first found and fixed three local `needless_borrow` lints; a retry on RCH
`vmi1153651` then failed in stale remote source with a `conv3d_backward_scalar_f64`
symbol that is absent from both local source and `origin/main`, so it is not
counted as a clean clippy proof for this landing.

## Negative evidence

The rejected prior `pinv` attempt showed why `Option`-returning QR kernels are
dangerous when the fallback is slow or missing. This `lstsq` pass avoids that
failure mode in the public API: QR is only the first attempt, and `None` falls
through to the already-merged SVD-based batched `lstsq` fallback.

The proof counts only the three full-rank overdetermined rows above. It does not
claim a speedup for rank-deficient or underdetermined batches; those are routed
to the general fallback.
