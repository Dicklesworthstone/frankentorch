# frankentorch-kgs4.158 — parallel strided (non-first-dim) FFT pass in tensor_fft_along_dim

Date: 2026-06-21
Agent: cc

## Lever

`tensor_fft_along_dim` parallelised only the CONTIGUOUS (last) dim case
(`par_chunks_mut(dim_size)`); the STRIDED non-last-dim path (gather→dft→scatter per
`(outer,inner)` lane) ran SERIALLY. n-D FFT (`fftn`/`ifftn`/`rfftn`/`irfftn`, which compose
per-dim via `tensor_fft_along_dim`) therefore ran every non-last dim serially. Each `outer`
block is a disjoint contiguous `dim_size*stride_inner` chunk, so for `stride_outer >= 2` the
outer blocks now distribute across the rayon pool (each gathers/dfts/scatters its strided
lanes in place). The per-lane DFT is compute-bound butterflies, so this is core-scaling.

## Correctness (bit-exact)

Disjoint outer blocks, identical per-lane DFT, identical gather/scatter index arithmetic →
bit-for-bit identical to the serial path. Same-process A/B reported IDENTICAL checksum
(`3.434956e6`). ft-api `--lib fft` 31 passed / 0 failed; ft-conformance green.

## Measurement (same-host, no-grad fftn, 32 threads, fixed-iter reused-session harness)

`example fftn_strided_ab`, complex128 input, fftn over all dims.

| Shape | serial | parallel (this) | internal |
| --- | ---: | ---: | --- |
| `[256,100,16]` (dim-1 bottleneck, stride_outer=256) | `86–94 ms` | `23.7–25.1 ms` | **~3.6x faster** |
| `[48,128,128]` (dim-0 bottleneck, stride_outer=1) | `571–579 ms` | `577 ms` | ~0 (bottleneck dim not covered) |

The win materialises whenever a non-first transform dim (stride_outer>=2) carries significant
work — true for most fftn. A dim-0-dominant shape sees no gain (dim-0 has stride_outer=1, still
serial; covering it needs a transpose-trick — separate follow-up).

## Verdict: KEEP — internal ~3.6x, PyTorch loss

vs PyTorch fftn `[256,100,16]`: PyTorch `0.64 ms/iter` vs FT `~24 ms` = ~37x slower. FT's FFT
is fundamentally pocketfft-walled (PyTorch uses mixed-radix O(N log N) even for non-power-of-2
sizes like 100; FT falls to an O(N²) DFT there) — an algorithmic/impl gap, NOT this parallel
pass. So this is an internal core-scaling win on FT's own n-D FFT (the FFT-vein's documented
"ship core-scaling internal wins" disposition), not a PyTorch win. Bit-exact, strictly removes
a serial bottleneck, no downside.

## Win/loss/neutral vs PyTorch (32t): `0W / 1L / 0N` (internal ~3.6x kept)

## Gates
- `cargo test -p ft-api --release --lib fft`: 31 passed, 0 failed.
- `cargo test -p ft-conformance --release`: green.
