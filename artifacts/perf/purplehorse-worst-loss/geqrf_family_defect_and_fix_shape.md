# The geqrf defect is a FAMILY defect — and the blocked kernel already computes what the fix needs

Two source findings that follow the certified 226x `geqrf` row (`48c0b7b7`), both cheap
to establish and both changing the scope of the fix.

## 1. `orgqr` has the same shape — this is a family defect, not one bad function

`tensor_orgqr` → `tensor_householder_product` applies reflectors **one at a time in a
reverse loop**:

```rust
for kk in (0..k).rev() {
    Self::apply_reflector_left(q, m, n, a_slice, n, kk, tau_slice[kk]);
}
```

and `apply_reflector_left` has the identical cache-hostile access pattern to
`geqrf_packed_f64`:

```rust
for j in 0..ncols {
    let mut w = c[kk * ncols + j];
    for i in (kk + 1)..rows {
        w += a_packed[i * a_cols + kk] * c[i * ncols + j];   // BOTH stride-n
    }
    c[kk * ncols + j] -= w;
    for i in (kk + 1)..rows {
        c[i * ncols + j] -= w * a_packed[i * a_cols + kk];   // BOTH stride-n
    }
}
```

Every inner loop indexes `[i * stride + col]` on a row-major matrix — one cache line
touched per useful element — and it runs once per reflector, so the whole build is
O(n³) of strided scalar work with no blocking and no GEMM.

So both halves of the QR primitive family in `ft-api` are private naive implementations
that never reach `ft-kernel-cpu`'s blocked path. **`orgqr`'s magnitude is NOT measured**
— it needs its own harness arm, since it consumes `(A, tau)` from a `geqrf` — and the
ledger's only figure for it (`torch@8 48 ms vs FT@8 ~86 ms`, ≈1.8x) is **cross-run and
at batched-tiny shapes**, a regime that never exercises this path. Given `geqrf`
measured 226x where the ledger's batched figures suggested nothing of the kind, that
1.8x should be treated as uninformative about the single-matrix case rather than as a
bound.

## 2. The fix is a re-route plus an exposure, not a rewrite

My bead flagged this as the open question: *"Confirm `qr_householder_panel_blocked`
retains or can return those before assuming this is a pure dispatch change — if it
discards them, the change is larger than a re-route."*

**It does not discard them.** The blocked forward pass computes both:

```rust
fn qr_factor_panel_leaf_f64(
    r_mat: &mut [f64], m: usize, n: usize,
    panel_start: usize, leaf_start: usize, leaf_end: usize, nb: usize,
    vmat: &mut [f64],     // <- the reflectors
    tau: &mut [f64],      // <- and tau
    tiny: f64,
)
```

and the entry point's own comment confirms it: *"Forward pass reduces R and stores each
panel's (V, T); Q is then built (m×qcols) by a reverse dorgqr."*

So `qr_householder_panel_blocked` already runs exactly the computation `geqrf` is,
and then throws away V and tau because its signature returns only Q (`-> Vec<f64>`).
The fix is to expose the forward pass — return the reduced `r_mat` plus `tau` and skip
the Q build — which is strictly *less* work than the existing blocked `qr` already does.

### What still has to be checked before calling it done

* **Packing layout.** LAPACK `geqrf` returns V packed below the diagonal of A with one
  `tau` per column. The blocked path keeps V in a separate `vmat` in compact-WY panel
  form. V and tau are recoverable in principle, but whether the layout matches what our
  own `orgqr`/`ormqr` consumers expect must be verified rather than assumed.
* **Bit-exactness is not free.** The blocked compact-WY path is documented as **not**
  bit-identical to the unblocked sweep above its threshold, admissible only under the
  ratified eig/SVD tolerance policy (`frankentorch-qgce4`). `geqrf`'s outputs feed
  `orgqr`/`ormqr`, so consumers must be checked under that policy exactly as the QR
  op's own blocked gate was.
* **`orgqr` needs the same treatment** and the same reverse-`dorgqr` already exists
  inside the blocked path — so fixing `geqrf` alone would leave the other half of the
  family on the naive route.

## Why this matters beyond the two ops

`geqrf` is 226x slower than torch while our own `qr` — which does strictly more work —
is 5.50x. That gap is not an algorithm problem and not a blocking problem; the blocked
code is *right there and already shipping for a neighbouring op*. The failure mode is a
**public API that bypasses its own optimised kernel**, and the reason it survived is
that the op had no live-torch in-invocation harness until this session.

That suggests a cheap sweep worth doing: for each `ft-api` linalg entry point, check
whether it calls into `ft-kernel-cpu` or carries a private loop. Any that carry private
loops are candidates for the same defect, and they can be found by reading rather than
by measuring.

## The sweep I recommended, run — three ops in the family, two candidates cleared by reading

I suggested checking every `ft-api` linalg entry point for whether it reaches
`ft-kernel-cpu` or carries a private loop, on the grounds that such defects are findable
by reading rather than by measuring. Ran it over 62 entry points. Most "no kernel call"
hits are thin aliases (`tensor_qr` → `tensor_linalg_qr`, `tensor_svd`, `tensor_det`,
`tensor_eigh`, …) with zero loops — correct as-is. What survives filtering to *no kernel
call **and** real loops with strided indexing*:

| entry point | verdict |
|---|---|
| `tensor_geqrf` | **DEFECT — measured 226x** (`48c0b7b7`), private `geqrf_packed_f64` |
| `tensor_householder_product` / `tensor_orgqr` | **DEFECT (source)** — per-reflector `apply_reflector_left` |
| `tensor_ormqr` | **DEFECT (source)** — same `apply_reflector_left` / `apply_reflector_right` |
| `tensor_linalg_solve_triangular` → `tensor_triangular_solve` | **CLEARED** |
| `tensor_linalg_lu_factor_ex` | **CLEARED** |
| `tensor_linalg_ldl_factor` | unresolved, weak signal |

### The two clears matter as much as the hits

`solve_triangular` looked like a strong candidate — no kernel call, loops, strided
indexing. Reading it showed the flagged loops were O(n) construction of a unit-diagonal
mask, and the real work delegates to `tensor_triangular_solve`, which **is** properly
gated:

```rust
if tri_blocked {
    x = ft_kernel_cpu::triangular_solve_blocked_contiguous_f64(...)
} else if tri_par { ... } else { ...scalar... }
```

Blocked kernel first, scalar only as fallback. Exactly the structure `geqrf` lacks.
`lu_factor_ex` likewise delegates to `tensor_linalg_lu_factor` (which does call the
kernel) and its loop is an O(n) `info` scan.

**A heuristic that flags private loops finds real defects and also finds wrappers.**
Both clears came from reading the code the heuristic pointed at, which is the whole
argument for this being a cheap sweep: three ops narrowed from 62, at no measurement
cost and with no window required.

### Scope of the family defect

`geqrf` (produce reflectors), `orgqr` (form Q from them) and `ormqr` (apply them to a
matrix) are the three LAPACK Householder primitives, and **all three** are private
per-reflector BLAS-2 loops in `ft-api` that never reach the blocked compact-WY kernel
already shipping for `tensor_linalg_qr`. Only `geqrf` has a measured number. The other
two need their own harness arms — both consume `(A, tau)` from a `geqrf`, so the fixture
must produce that outside the timed region.

The ledger's figures for `orgqr` (~1.8x) and `ormqr` (LOSS) are cross-run at
batched-tiny shapes and should not be read as bounds on the single-matrix path — the
same reasoning that made `geqrf`'s 226x a surprise.

## RESOLVED: the packed-V layout question — the fix is a re-route plus an O(n²) scatter

The bead left one thing genuinely open: *"whether the layout matches what our own
`orgqr`/`ormqr` consumers expect must be verified rather than assumed."* Verified.

`qr_householder_panel_blocked_profiled` accumulates every panel:

```rust
let mut panels: Vec<(usize, Vec<f64>, Vec<f64>)> = Vec::new();   // (start, vmat, tau)
...
let mut vmat = vec![0.0f64; m * nb];
let mut tau  = vec![0.0f64; nb];
qr_factor_panel_recursive_f64(r_mat, m, n, p, 0, nb, nb, &mut vmat, &mut tau, tiny);
```

So **`V` and `tau` for every panel survive the whole forward pass** — they are not
consumed and dropped panel-by-panel, they are collected. Three consequences:

1. **`tau` needs no conversion at all.** Each panel's `tau` is a flat per-column vector
   and the panels are processed in column order, so concatenating them is exactly
   LAPACK's `tau`.
2. **`V` needs an O(n²) scatter, not a re-derivation.** A panel's `vmat` is `m × nb` with
   the standard staircase — column `j` zero above row `p+j`, unit at `p+j`, packed below.
   LAPACK's `geqrf` puts that same reflector in `A[p+j+1.., p+j]`. So the conversion is
   `a[i*n + (p+j)] = vmat[i*nb + j]` for `i > p+j`, per panel. That is O(n²) work against
   an O(n³) factorisation — free.
3. **`R` is already in place.** The forward pass reduces `r_mat` in situ, which is where
   `geqrf` wants it.

**So `geqrf` = the existing blocked forward pass, minus the Q build, plus an O(n²)
scatter.** It is strictly *less* work than the blocked `qr` already performs, which is
consistent with the measurement: our `qr` (40.707 ms at n=512) is 13.7x faster than our
`geqrf` (559.481 ms) while doing more.

What remains genuinely open is only the **bit-exactness** question, unchanged: the
blocked compact-WY path is documented as not bit-identical to the unblocked sweep above
its threshold, and `geqrf`'s outputs feed `orgqr`/`ormqr`, so the change lands under the
ratified eig/SVD tolerance policy (`frankentorch-qgce4`) and its consumers need checking
exactly as the QR op's own blocked gate did.

`orgqr` gets the same treatment for free: the blocked path already contains the reverse
`dorgqr` that builds Q from `(V, T)`, so once `geqrf` returns the packed reflectors, the
Q-formation side has an existing implementation to call rather than the per-reflector
loop it uses today.
