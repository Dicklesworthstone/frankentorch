# frankentorch-66pe: cumprod outer-block Rayon rejection

## Target

- Hotspot: `tensor_cumprod` over `[8192, 1024]` along `dim=1`.
- Candidate lever: fan independent outer blocks over Rayon for f64/f32 cumprod.
- Bead: `frankentorch-66pe`.

## Benchmark Evidence

Command:

```bash
rch exec -- env RAYON_NUM_THREADS=1 cargo bench -p ft-api --bench ops_bench -- cumprod/nograd_8192x1024_dim1 --warm-up-time 1 --measurement-time 5 --sample-size 20
rch exec -- cargo bench -p ft-api --bench ops_bench -- cumprod/nograd_8192x1024_dim1 --warm-up-time 1 --measurement-time 5 --sample-size 20
```

Comparable same-worker pair on `ts1`:

| Run | Worker | Time |
| --- | --- | --- |
| Serial baseline, `RAYON_NUM_THREADS=1` | `ts1` | `[41.271 ms 42.202 ms 43.401 ms]` |
| Default-thread candidate | `ts1` | `[74.018 ms 77.261 ms 81.307 ms]` |

Cross-worker non-comparable first baseline:

- `vmi1149989`: `[65.250 ms 71.580 ms 77.754 ms]`

Verdict: rejected. Same-worker median regressed `42.202 ms -> 77.261 ms` (`0.55x`, 1.83x slower), below Score >= 2.0.

## Isomorphism Proof

- Ordering preserved: yes in candidate; each lane retained identical d-order multiplication.
- Tie-breaking unchanged: N/A.
- Floating-point: bit-exact per lane; no reassociation inside each row.
- RNG seeds: N/A for kernel; benchmark input deterministic.
- Focused proof: `rch exec -- cargo test -p ft-kernel-cpu cumprod_parallel_matches_serial_bit_exact` passed on `ts1`.
- Golden SHA-256: no runtime change kept. Evidence log checksums are recorded in `evidence.sha256`.

## Closeout

The candidate source and benchmark hunk were removed. Next deeper primitive should avoid Rayon per-call scheduling for scan kernels; likely target is a persistent/chunked scan primitive or API-level batching where the scheduling cost is amortized across multiple scan rows and ops.
