# frankentorch-c6uw pass 8 probe: rank-2 split / parallel packed update rejection

## Target

Follow-up probe after the pass-8 direct-reflector keep. The intended lever was
to split the Householder packed rank-2 update into a sequential `u = e - hh*v`
prepass and a row-range update helper, then use Rayon only for large packed
trailing updates.

## Result

Rejected. The source hunk was removed.

The deterministic payload from `eigh_golden` stayed bit-identical:

- before payload SHA-256: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`
- after payload SHA-256: `43e8c0e7c868d54d8ed62fd4da30d4c2efe3b1889e9c350c50f5cbf7539add16`

Focused tests passed while the hunk was present:

- `eigvalsh_matches_eigh`
- `eigh_tred2_tql2_orthonormal_and_reconstructs_24x24`

Same-worker `ts2` after row with the hunk present:

- `eigh_f64_256x256`: `[38.093 ms 38.510 ms 38.905 ms]`
- `eigvalsh_f64_256x256`: `[33.563 ms 33.852 ms 34.207 ms]`

This is a hard regression relative to both the current direct-reflector keep
and the existing baseline rows, so it fails the Score gate.

## Next Primitive

Do not retry row fan-out or packed row-split micro-levers. Continue with
`frankentorch-rd1s`: a real safe-Rust LAPACK-class blocked `dsytrd` panel /
compact-WY / BLAS-3 trailing rank-2k update and tridiagonal D&C/secular merge
with an explicit FP reassociation ledger and exact EISPACK fallback.
