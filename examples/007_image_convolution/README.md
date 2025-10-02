# 007: Image Convolution - GPU Performance Sweet Spot

Demonstrates OpenCL's optimal use case: image processing with high arithmetic intensity.

## Purpose

Show **when, where, and why OpenCL becomes the most efficient choice** by comparing implementations across varying image sizes and convolution kernel sizes.

## What It Does

Applies Gaussian blur filter (2D convolution) using multiple approaches:
1. **Serial C++**: Baseline single-threaded implementation
2. **OpenMP**: Multi-threaded CPU parallelization
3. **OpenCL (simple)**: Straightforward GPU kernel
4. **OpenCL (local memory)**: Optimized with tile caching
5. **OpenCL (separable)**: Decomposed 2D → two 1D passes

Tests across:
- Image sizes: 512×512, 1024×1024, 2048×2048, 4096×4096
- Kernel sizes: 3×3, 5×5, 7×7, 11×11, 15×15

## Key Results

### Small Kernels (3×3) - OpenMP Competitive

```
512×512, 3×3:
Serial:    2.85ms (baseline)
OpenMP:    2.05ms (1.4x)
GPU:       0.88ms (3.2x)
```

Small kernels have low arithmetic intensity - OpenMP is sufficient.

### Large Kernels (15×15) - OpenCL Dominates

```
4096×4096, 15×15:
Serial C++:                 3369.6ms (baseline)
OpenMP:                      526.4ms (6.4x)
NVIDIA GPU (separable):       28.7ms (117x) 
Intel CPU OpenCL (separable): 22.5ms (150x) ⭐
```

Large kernels provide high arithmetic intensity - OpenCL provides 20-25x better speedup than OpenMP.

## Performance Hierarchy

**For large workloads (4096×4096, 15×15 kernel):**

| Rank | Implementation | Time | Speedup |
|------|----------------|------|---------|
| 1st | Intel CPU (separable) | 22.5ms | **150x** |
| 2nd | Intel CPU (local) | 27.6ms | 122x |
| 3rd | NVIDIA GPU (separable) | 28.7ms | 117x |
| 4th | NVIDIA GPU (local) | 33.3ms | 101x |
| 5th | Intel UHD (separable) | 41.1ms | 82x |

OpenMP plateaus at 6.4x due to memory bandwidth saturation across 16 cores.

## Why OpenCL Wins Here

**Image convolution is GPU-optimal because:**

1. **Perfect data parallelism**: Every pixel computed independently
2. **High arithmetic intensity**: 225 operations per pixel (15×15 kernel) vs ~3 memory reads
3. **Data reuse patterns**: Neighboring pixels share overlapping input regions
4. **Regular memory access**: Predictable patterns enable memory coalescing

### Arithmetic Intensity Comparison

```
Operation              Ops/Pixel    GPU Advantage
-----------------------------------------------------
Vector addition        1            ✗ GPU loses
Matrix-vector (4K)     4096         ~ Break-even
Matrix multiply (2K)   2048         ✓ GPU wins (155x)
Convolution 3×3        9            ~ Marginal (3x)
Convolution 15×15      225          ✓✓ GPU dominates (150x)
```

As operations-per-memory-access increases, GPU advantage grows exponentially.

## Optimization Techniques

### 1. Local Memory Tiling

Caches frequently accessed data in GPU's fast local memory:

```c
__local float tile[TILE_SIZE + 2*HALO][TILE_SIZE + 2*HALO];
```

**Impact:**
- NVIDIA: 176ms → 33ms (5.3x improvement)
- Reduces global memory accesses by ~16x

### 2. Separable Convolution

Decomposes 2D convolution into two 1D passes:
- Complexity: O(n²×k²) → O(n²×2k)
- For 15×15 kernel: 225 ops → 30 ops (7.5x reduction)

**Impact:**
- Best speedups across all devices
- Intel CPU OpenCL: 150x (vs 6x OpenMP)

### 3. Why Intel CPU Beats NVIDIA GPU

At the largest test, Intel CPU OpenCL outperforms NVIDIA discrete GPU:

**Reasons:**
1. **No PCIe overhead** - Data never leaves CPU memory
2. **Cache efficiency** - 64MB problem fits in 45MB L3 cache
3. **Separable optimization** - Sequential 1D passes maximize cache hits
4. **Memory locality** - All 16 cores share L3 cache

For this specific workload, cache locality trumps raw GPU parallelism.

## Building

```cmd
build.bat
```

## When to Use OpenCL for Image Processing

Based on comprehensive testing:

| Kernel Size | Best Choice | Why |
|-------------|-------------|-----|
| 3×3, 5×5 | OpenMP | Low arithmetic intensity, overhead dominates |
| 7×7 | OpenCL (marginal) | Breakeven point ~30x speedup |
| 11×11+ | OpenCL (clear winner) | High arithmetic intensity, 50-150x speedup |

**OpenCL becomes optimal when:**
- Kernel size ≥ 7×7
- Image size ≥ 1024×1024
- Separable decomposition possible
- Tiling/local memory can be exploited

## Performance Scaling

Speedup increases with both image size and kernel size:

```
NVIDIA GPU (separable) speedup:
512×512:
  3×3:    3x
  15×15:  102x

4096×4096:
  3×3:    7x
  15×15:  117x
```

Larger problems amortize GPU overhead and exploit parallelism better.

## Common Pitfalls

### Mistake: Using `kernel` as Parameter Name

```c
// ✗ WRONG - 'kernel' is OpenCL reserved keyword
__kernel void convolve(__constant float* kernel, ...)

// ✓ CORRECT
__kernel void convolve(__constant float* filter, ...)
```

This caused all devices to fail compilation initially.

## Key Takeaways

1. **Image processing is OpenCL's sweet spot** - this is what GPUs were designed for
2. **Arithmetic intensity matters more than parallelism** - simple parallel operations lose to serial
3. **Optimization techniques are essential** - naive GPU code provides minimal benefit
4. **CPU OpenCL can beat GPU** - when data locality matters more than raw throughput
5. **Problem size determines winner** - small images with small kernels favor OpenMP

## Real-World Applications

This pattern applies to:
- Image filtering (blur, sharpen, edge detection)
- Computer vision (feature extraction, template matching)
- Video processing (real-time effects)
- Medical imaging (CT/MRI reconstruction)
- Scientific visualization (volume rendering)

All benefit from OpenCL when kernel sizes and image resolutions are large.

## Next Steps

To further improve performance:
- Implement FFT-based convolution for very large kernels (O(n² log n))
- Use vendor-specific libraries (cuDNN, clBLAS) for production
- Explore image formats (packed RGBA vs planar) impact
- Test with real-world images vs synthetic data