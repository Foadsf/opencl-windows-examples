# 002: Vector Addition Performance Comparison

Compares CPU vs GPU performance for vector addition across all available devices.

## Purpose

Demonstrate that **simple operations don't benefit from GPU acceleration** due to memory transfer overhead.

## What It Does

Performs element-wise vector addition (`C[i] = A[i] + B[i]`) using:
1. Serial C++ (single-threaded baseline)
2. OpenCL on each available device (NVIDIA GPU, Intel GPU, Intel CPU)

## Results (10M elements)

```
Vector size: 10000000 elements
Memory per vector: 38.147 MB

Serial C++:                    6.12 ms    1.00x (baseline)
OpenCL: NVIDIA RTX A2000:     24.23 ms    0.25x (SLOWER!)
OpenCL: Intel UHD Graphics:   10.09 ms    0.61x (SLOWER!)
OpenCL: Intel CPU:             7.72 ms    0.79x (SLOWER!)
```

## Why is GPU Slower?

Vector addition is **memory-bound**, not compute-bound. Each element requires:
- 1 addition operation (trivial)
- 2 memory reads + 1 memory write (expensive)

**GPU overhead dominates:**
1. Copy 38 MB to GPU memory (PCIe transfer)
2. Perform trivial computation
3. Copy 38 MB result back to CPU

Total transfer time exceeds the CPU's cache-optimized serial execution.

## Key Lesson

**Not all parallel operations benefit from GPUs.** Memory-bound operations with low computational intensity are better suited for CPU execution.

## Building

```cmd
build.bat
```

## Next Steps

- `003_breakeven_analysis`: Find the crossover point where GPUs become worthwhile
- `006_matrix_multiply`: See compute-intensive operations where GPUs dominate