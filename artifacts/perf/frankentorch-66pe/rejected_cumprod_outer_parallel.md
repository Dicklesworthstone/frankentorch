# frankentorch-66pe: cumprod outer-block Rayon probe rejected

## Target

- Bead: `frankentorch-66pe`
- Surface: `ft-kernel-cpu` cumprod scan kernel, exercised through
  `ft-api` `ops_bench` as `cumprod/nograd_8192x1024_dim1`
- Candidate lever: mirror the landed `cumsum` outer-block Rayon fan-out for
  `cumprod_tensor_contiguous_{f64,f32}` while preserving each lane's
  multiplication order.

## Proof

Focused bit-exact test:

```bash
rch exec -- cargo test -p ft-kernel-cpu cumprod_parallel_matches_serial_bit_exact
```

Result on `ts1`: `1 passed; 0 failed`.

Isomorphism notes for the candidate:

- Ordering preserved: yes, output indexing and row-major order unchanged.
- Tie-breaking unchanged: N/A.
- Floating-point: per-lane multiplication order unchanged, no reassociation
  within a lane; bit-exact test passed.
- RNG: N/A.

## Benchmark

Same-worker `ts1` evidence:

- Serial control (`RAYON_NUM_THREADS=1`):
  `[41.271 ms 42.202 ms 43.401 ms]`
- Default Rayon candidate:
  `[74.018 ms 77.261 ms 81.307 ms]`

Additional cross-worker/default-thread signal was also unfavorable:

- `vmi1156319` default Rayon candidate:
  `[89.997 ms 93.566 ms 97.325 ms]`

## Verdict

Rejected. The candidate is behavior-preserving but slower than the same-worker
serial control, so it fails the Score >= 2.0 keep gate. Source and benchmark
hunks were removed; only the rejection evidence and bead closeout should land.

Next scan-kernel route: do not assume the `cumsum` outer-block fan-out
generalizes to product scans. Re-profile `cummax`/`cummin`/`logcumsumexp` or
move to a deeper scan primitive with chunk-local summaries plus deterministic
combine only if it has a profile-backed target and a strict FP/order ledger.
