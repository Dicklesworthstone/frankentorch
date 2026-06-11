# Pass 3 - Shift-Source Profile Scaffold

Bead: `frankentorch-npxbw`

## One Lever

Implemented a hidden diagnostic scaffold around the existing scalar Francis double-shift QR path.

Added scaffold API:
- `EigFrancisShiftSample`
- `EigFrancisProfile`
- `EigFrancisProfileResult`
- `eig_francis_profile_f64(data, meta, want_vectors)`

The production `eig_contiguous_f64` and `eigvals_contiguous_f64` dispatch remains unchanged: both still route through `eig_impl`, which still calls the non-profiled `eig_francis_schur` wrapper. The wrapper uses `FrancisTraceDisabled`; the diagnostic entry point uses the traced helper.

No multishift QR, AED retry, or public dispatch change was made. The only non-diagnostic edits outside the Francis helper were clippy-only `needless_range_loop` rewrites in shared `eigh_tred2_*` accumulation loops; they keep the same ascending `k` order and value reads.

## Profile Fields

The hidden profile records:
- matrix dimension `n`
- total Francis sweeps
- 1x1 deflations
- 2x2 deflations
- fallback deflations
- global max-iteration fallback exhaustions
- exceptional shifts
- max active window width
- bounded active-window samples
- bounded shift samples with `(active_first, active_last, iteration_in_window, accumulated_shift, x, y, w, exceptional)`
- bounded selected bulge-start rows

## Isomorphism Proof

- Ordering preserved: yes. The diagnostic path mirrors `eig_impl` setup/reduction and calls the same traced Francis helper; eigenvalue slots are written by the existing bottom-up deflation order.
- Tie-breaking unchanged: yes. No sorting, reordering, or block reclassification was added.
- Floating-point behavior: production path unchanged. Diagnostic path adds trace writes only after existing scalar shift values are formed and does not feed trace data back into arithmetic.
- RNG: unchanged; no RNG used.
- Golden output: strict `eigvals_golden` output SHA-256 matched `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.

## Profile Probe

`rch exec -- cargo run --release -q -p ft-kernel-cpu --example eig_timing_probe`

Remote worker: `vmi1152480`

Rows from `pass3_eig_timing_probe_final.log`:
- `n=128`: `eigvals=4.96ms`, `eig=8.19ms`, profile `sweeps=173 defl1=28 defl2=50 fallback=0 exceptional=0 max_width=128 samples=173 truncated=false`
- `n=256`: `eigvals=32.04ms`, `eig=55.20ms`, profile `sweeps=319 defl1=14 defl2=121 fallback=0 exceptional=0 max_width=256 samples=319 truncated=false`
- `n=512`: `eigvals=403.22ms`, `eig=591.06ms`, profile `sweeps=583 defl1=10 defl2=251 fallback=0 exceptional=0 max_width=512 samples=583 truncated=false`
- `n=1024`: `eigvals=3022.04ms`, `eig=5078.17ms`, profile `sweeps=1132 defl1=18 defl2=503 fallback=0 exceptional=0 max_width=1024 samples=1132 truncated=false`

These rows are profile-shape evidence for the next source lever. They are not keep/reject proof because pass 3 intentionally does not change production dispatch.

## Validation

- `rch exec -- cargo test -p ft-kernel-cpu --lib eig`: passed on `vmi1149989`; 21 eig/eigh-focused tests passed, including `eig_francis_profile_matches_eigvals_bit_exact`, `eigvals_matches_eig`, `eigvals_companion_complex_roots`, and `eig_parallel_schur_vector_update_matches_single_thread_bit_exact`.
- `rch exec -- cargo run -q -p ft-kernel-cpu --example eigvals_golden`: passed on `vmi1149989`; strict SHA-256 matched pass 1 exactly: `24ed0e24afc1b41d3b23198f60fc1d06727374bf3551c026941a25785b7c9725`.
- `rch exec -- cargo check -p ft-kernel-cpu --lib --examples --benches`: passed warning-free on `vmi1153651`.
- `rch exec -- cargo clippy -p ft-kernel-cpu --lib --examples --benches -- -D warnings`: passed on `vmi1149989`.
- `rch exec -- cargo fmt -p ft-kernel-cpu --check`: passed.
- `ubs crates/ft-kernel-cpu/src/lib.rs crates/ft-kernel-cpu/examples/eig_timing_probe.rs`: exit 0; no critical issues. UBS reported the existing broad warning corpus in these files.

## Verdict

Productive implementation pass. The shift-source/profile scaffold is available for Pass 4 direct small-bulge or BLAS-3 far-update work, with production eig/eigvals dispatch behavior preserved.
