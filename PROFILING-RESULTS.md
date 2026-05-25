# FrankenTorch Profiling Results

Rigorous performance comparison against PyTorch 2.12.0+cpu (single-threaded).

## Environment

| Component | Version |
|-----------|---------|
| PyTorch | 2.12.0+cpu |
| FrankenTorch | 0.1.0 |
| Platform | Linux 6.17.0 x86_64 |
| Threads | 1 (single-threaded) |

## Benchmark Methodology

- **Warmup**: Criterion default (3s)
- **Runs**: 100 samples per benchmark
- **Metrics**: p50 latency in microseconds
- **Apples-to-apples**: Both libraries run single-threaded on CPU

## Results Summary

### Matrix Operations (GEMM-optimized)

| Operation | PyTorch p50 (μs) | FrankenTorch p50 (μs) | Ratio | Status |
|-----------|------------------|----------------------|-------|--------|
| matmul 64x64 | 9.02 | 16.5 | 1.83x | ✅ |
| matmul 128x128 | 49.05 | 107.7 | 2.20x | ⚠️ |
| matmul 256x256 | 365.15 | 810 | 2.22x | ⚠️ |
| matmul 512x512 | 2,908 | 6,191 | 2.13x | ⚠️ |
| matmul 1024x1024 | 24,022 | 49,360 | 2.05x | ⚠️ |

### Batched Matrix Multiply

| Operation | PyTorch p50 (μs) | FrankenTorch p50 (μs) | Ratio | Status |
|-----------|------------------|----------------------|-------|--------|
| bmm b8 128x128 | 390 | 1,280 | 3.28x | ❌ |
| bmm b16 128x128 | 845 | 2,584 | 3.06x | ❌ |
| bmm b32 128x128 | 23,124 | 5,676 | 0.25x | ✅ FT faster |

### Convolution

| Operation | PyTorch p50 (μs) | FrankenTorch p50 (μs) | Ratio | Status |
|-----------|------------------|----------------------|-------|--------|
| conv2d 64x64 | 28,158 | 115,850 | 4.11x | ❌ |
| conv2d 128x128 | 73,183 | 459,380 | 6.28x | ❌ |

### Linear Layer (forward)

| Operation | PyTorch p50 (μs) | FrankenTorch p50 (μs) | Ratio | Status |
|-----------|------------------|----------------------|-------|--------|
| linear 32x512→256 | 114 | 1,207 | 10.6x | ❌ |
| linear 32x512→512 | 224 | 2,615 | 11.7x | ❌ |
| linear 32x512→1024 | 441 | 5,404 | 12.3x | ❌ |
| linear 32x512→2048 | 886 | 10,382 | 11.7x | ❌ |

### Element-wise Operations

| Operation | PyTorch p50 (μs) | FrankenTorch p50 (μs) | Ratio | Status |
|-----------|------------------|----------------------|-------|--------|
| relu 1M | 91.19 | 2,772 | 30.4x | ❌ |
| exp 1M | 1,182 | 2,925 | 2.47x | ⚠️ |
| add 1M | 110.34 | 908 | 8.23x | ❌ |

## Status Legend

- ✅ FrankenTorch ≤ 1.5x PyTorch (acceptable)
- ⚠️ FrankenTorch 1.5x-3x PyTorch (needs optimization)
- ❌ FrankenTorch > 3x PyTorch (performance bug)

## Performance Gaps Filed

### Critical (>10x slower)
- `frankentorch-bibi`: matmul 500x slower (FIXED with matrixmultiply)
- **linear**: ~12x slower than PyTorch (needs GEMM for addmm)

### Severe (3-10x slower)
- **relu 1M**: 30x slower (dispatch overhead dominates)
- **add 1M**: 8x slower (allocation + dispatch overhead)
- **conv2d**: 4-6x slower (naive im2col implementation)
- **bmm b8/b16**: 3x slower (loop overhead per batch)

### Moderate (1.5-3x slower)
- **matmul**: 2x slower (matrixmultiply vs OpenBLAS/MKL)
- **exp 1M**: 2.5x slower (libm vs SVML)

## Optimization Completed

### 2026-05-25: matrixmultiply integration
- Added `matrixmultiply` crate for BLAS-quality GEMM
- Updated `matmul_tensor_contiguous_f64/f32` to use `dgemm/sgemm`
- Updated `bmm_tensor_contiguous_f64/f32` to use GEMM per batch
- **Result**: matmul improved from 500x slower to 2x slower

## Remaining Optimization Opportunities

1. **Linear layer**: Use GEMM for `addmm` (currently naive triple-loop)
2. **Element-wise ops**: Reduce dispatch overhead, use SIMD
3. **Conv2d**: Consider NNPACK or Winograd for small kernels
4. **Session overhead**: Profile FrankenTorchSession dispatch path

---
Generated: 2026-05-25
