# 006: Matrix Multiplication - Where GPUs Shine

Demonstrates compute-intensive operations where GPUs provide massive speedups (100x+).

## Purpose

Show the true power of GPU acceleration with O(n³) computational complexity.

## Operation: Matrix-Matrix Multiplication

Compute `C = A × B` where all matrices are N×N.

- Complexity: **O(n³)** operations (2×N³ multiply-adds)
- Memory access: O(n²) reads/writes
- **High compute-to-memory ratio**: Perfect for GPUs

## Implementations

1. **Serial C++**: Naive triple nested loop
2. **C++ std::execution::par**: Parallel outer loops
3. **OpenMP**: Parallel double loop (collapse unsupported in MSVC)
4. **OpenCL (simple)**: Straightforward GPU kernel
5. **OpenCL (tiled)**: Optimized with local memory

## Results: 2048×2048 Matrices

```
Implementation                     Time (ms)    GFLOPS    Speedup
------------------------------------------------------------------
Serial C++                         16,306.05      1.05      1.00x
C++ std::execution::par             6,146.84      2.79      2.65x
OpenMP                              5,242.17      3.28      3.11x

OpenCL: NVIDIA (simple)             1,002.92     17.13     16.26x
OpenCL: NVIDIA (tiled)                105.13    163.42    155.11x ⭐

OpenCL: Intel UHD (simple)          3,563.28      4.82      4.58x
OpenCL: Intel UHD (tiled)             598.97     28.68     27.22x

OpenCL: Intel CPU (simple)            860.46     19.97     18.95x
OpenCL: Intel CPU (tiled)             562.05     30.57     29.01x
```

## Critical Observations

### 1. GPU Dominates at Scale

**NVIDIA GPU achieves 155x speedup** - processing 163 GFLOPS vs CPU's 3.3 GFLOPS. This is the performance gap we expect from GPUs on compute-intensive workloads.

### 2. Tiling is Essential

Local memory optimization provides dramatic improvements:

- NVIDIA: 1003ms → **105ms** (10x faster)
- Intel UHD: 3563ms → **599ms** (6x faster)

Tiling reduces global memory access by caching data in fast local/shared memory.

### 3. Speedup Grows with Problem Size

| Matrix Size | NVIDIA Speedup (Tiled) |
|-------------|------------------------|
| 256×256     | 15.4x                  |
| 512×512     | 37.3x                  |
| 1024×1024   | 83.1x                  |
| 2048×2048   | **155.1x**             |

GPU advantage increases exponentially with scale.

### 4. OpenMP Plateaus

At 2048×2048, OpenMP only achieves 3.1x speedup. Memory bandwidth saturation limits further scaling across 16 cores competing for shared L3 cache.

## Why This Works (vs Examples 2-5)

**Matrix multiplication has favorable compute-to-memory ratio:**

- 2048×2048: **17.2 billion operations**, only 33MB data transfer
- Ratio: ~520 operations per byte transferred
- GPU can hide memory latency with massive parallelism

**Previous examples (vector ops, matvec) were memory-bound:**

- Too few operations per byte
- GPU idle waiting for memory
- PCIe transfer overhead dominates

## Kernel Optimization: Tiling

The tiled kernel uses 16×16 work-group tiles with local memory:

```c
__local float A_tile[16][16];
__local float B_tile[16][16];
```

**Benefits:**
1. Reduces global memory accesses by 16x
2. Exploits GPU cache hierarchy
3. Improves memory coalescing
4. Amortizes transfer overhead

## Building

```cmd
build.bat
```

## Performance Tips

For even better performance, consider:
- Larger tile sizes (32×32) on high-end GPUs
- Register blocking
- Strassen's algorithm for huge matrices
- cuBLAS/clBLAS libraries for production code

## Key Takeaway

**GPUs excel when:**
- Compute intensity is high (O(n³) or higher)
- Problem size is large (> 1024×1024)
- Data reuse is possible (tiling)
- Memory bandwidth isn't the bottleneck

Matrix multiplication is the canonical example of GPU-suitable computation.