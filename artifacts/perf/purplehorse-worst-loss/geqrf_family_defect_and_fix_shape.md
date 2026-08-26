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
