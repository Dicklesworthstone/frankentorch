# Negative evidence ledger: batched lstsq QR

## Kept

- Full-rank overdetermined no-grad f64 `lstsq` batches use the QR-parallel path.
- Final measured ratios vs PyTorch CPU local sidecar:
  - `B=100000 m=8 n=4 rhs=2`: `14.27x` faster.
  - `B=20000 m=16 n=8 rhs=2`: `4.79x` faster.
  - `B=8000 m=32 n=16 rhs=2`: `1.82x` faster.
- Checksums matched PyTorch on every row.

## Not claimed

- Rank-deficient batched `lstsq` is not counted as a QR win.
- Underdetermined batched `lstsq` is not counted as a QR win.
- RCH worker-local PyTorch was unavailable (`ModuleNotFoundError: No module named 'torch'`), so ratio evidence uses the local PyTorch CPU sidecar.

## Reverted or avoided

- No fallback-removal was attempted.
- No rank-deficient error path was shipped. QR `None` falls through to the existing SVD batched `lstsq` implementation.
- No new worktree or `.scratch` directory was created.

## Next target

Continue the batched-linalg class where the profile still loses or has missing
coverage: `svdvals` f32, tiny-k `svd`, and f32 mirrors of `eigvals`/`eig`.
