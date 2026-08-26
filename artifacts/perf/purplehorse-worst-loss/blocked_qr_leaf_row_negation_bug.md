# The blocked QR leaf negated a row on every SQUARE input — a pre-existing bug, found by routing the family's only sign-sensitive checksum

**`tensor_linalg_qr` has dispatched to the blocked compact-WY leaf at `m >= 128 && k >= 16`
since long before this campaign. For any SQUARE input it emitted a spurious `tau = 2`
reflector on the last column — `H = I - 2 e_j e_jᵀ`, i.e. a ROW NEGATION. Fixed in
`380699a2`. 66 tests green across both crates after the fix.**

```
square tau[95]: blocked 1.9999999999999998, naive 0     (96x96, before)
test result: ok. 2 passed; 0 failed                      (after)
```

## The defect

LAPACK `dlarfg` emits NO reflector (`tau = 0`) when a column's BELOW-diagonal part is
already zero: the column is upper-triangular, nothing needs eliminating, however large the
diagonal. The leaf tested the WHOLE column norm instead:

```rust
let norm_v = nrm2.sqrt();       // nrm2 spans the whole column, diagonal INCLUDED
if norm_v < tiny { continue; }  // misses "below-diagonal already zero"
```

so on a column whose below-diagonal is empty it built a reflector with `v = e_j`,
`tau_L = 2`, which negates row j.

**Only reachable when a column's below-diagonal is empty — the LAST column of a SQUARE
matrix.** A tall matrix (m > n) never has one.

The fix leaves `nrm2`'s accumulation order EXACTLY as it was (diagonal summed first), so
every non-skipped column is bit-identical to before; only the skip decision changed.

## Why nothing caught it: the whole family's checksums are SIGN-BLIND

| lane | checksum | sees a row negation? |
|---|---|---|
| `qr`, `geqrf` | `\|diag(R)\|` | NO |
| `orgqr` | `\|Q\|` elementwise | NO — **by design**, "Q is unique only up to COLUMN SIGNS" |
| `ormqr` | `\|Q C\|` | **YES** |

The `orgqr` lane's blindness is defensible in isolation and catastrophic in aggregate: Q
really is unique up to column signs, so taking `|Q|` is the right call for THAT lane — but
it meant the family had no sign-sensitive check anywhere until `ormqr` was routed into the
same kernel.

`ormqr` read **6.13e-4 / 1.53e-3 MISMATCH** where the per-reflector path it replaced read
**1.54e-12 MATCH**. That is the entire reason this was found.

## My own tests were blind twice over

Both blind spots were in tests I wrote while explicitly reasoning about what could break:

1. **All fixtures TALL** (M=160 > N) — so none had a column with an empty below-diagonal.
2. **Two were SINGLE-PANEL** (N=32 against `nb_block = 32`) — and with one panel, REVERSE
   and FORWARD panel traversal are IDENTICAL. The tests written to verify panel ordering
   were structurally incapable of detecting a panel-ordering bug.

A third: my first `ormqr` test built its reference with `orgqr_blocked_f64`, which shares
`householder_panels_from_packed_f64` with the code under test — a bug in the shared panel
builder would have cancelled out and passed. It had to be rewritten against a naive
per-reflector reference. That flaw surfaced only because the test PANICKED for an unrelated
reason (`orgqr_blocked_f64`'s `n` is both the packed row stride and the output width, so
asking it for a full m x m Q reads `packed` with the wrong stride).

**`Q R == A` is necessary but NOT sufficient.** Any valid QR satisfies it, including one
whose reflectors differ in sign from LAPACK's. To claim a re-route preserves behaviour,
compare ELEMENTWISE against the implementation being replaced — which is what
`blocked_geqrf_matches_naive_*` now does, on square AND tall, on packed AND tau.

## Fixture checklist for any blocked/panelled kernel

* at least one SQUARE case (exercises the empty-below-diagonal column)
* at least one MULTI-PANEL case (`k > nb_block`, so ordering is observable)
* at least one TALL case
* reference built INDEPENDENTLY of the code under test

## Consequence for `qr`

For square inputs, `qr`'s Q no longer carries a negated last column. That was defensible
under "Q is unique up to column signs" and is NOT LAPACK's convention — anything comparing
our Q elementwise against torch's would have disagreed, and `qr`'s own `|diag(R)|` checksum
could never have shown it.

## Honest cost

This was found because I shipped `984cb985` asserting `ormqr` correct on tests that could
not fail the way the code could break, and because I reported `geqrf` as "validated" on a
`|diag(R)|` checksum. The regression I introduced is what exposed the older bug underneath
it. The trade was worth making, but the tests should have been built this way first.
