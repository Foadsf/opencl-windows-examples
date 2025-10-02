# 005: Parallelization Technologies Comparison

Compares five parallelization approaches: Serial C++, C++17 std::execution, OpenMP, and OpenCL across three devices.

## Purpose

Understand which parallelization technology is best for **matrix-vector multiplication** (moderate compute intensity).

## Technologies Tested

1. **Serial C++**: Single-threaded baseline
2. **C++ std::execution::par**: C++17 standard library parallelism
3. **OpenMP**: Industry-standard shared-memory parallelism
4. **OpenCL** (3 devices): NVIDIA GPU, Intel UHD GPU, Intel CPU

## Operation: Matrix-Vector Multiplication

Compute `y = A × x` where A is M×N matrix, x is N-vector.

- Complexity: O(M×N) operations
- Memory access: O(M×N) reads

Still relatively memory-bound, not ideal for GPUs.

## Results (4096×4096 matrix)

```
Implementation                  Time (ms)    Speedup
-----------------------------------------------------
Serial C++                        16.39      1.00x
C++ std::execution::par            1.73      9.47x
OpenMP                             1.50     10.92x
OpenCL: NVIDIA RTX A2000          23.38      0.70x (SLOWER)
OpenCL: Intel UHD Graphics        63.50      0.26x (SLOWER)
OpenCL: Intel CPU                  4.37      3.75x
```

## Key Findings

1. **OpenMP wins decisively**: 11x speedup with minimal overhead
2. **C++ std::execution::par competitive**: 9.5x speedup, standard library
3. **GPUs still slower**: Memory bandwidth bottleneck persists
4. **OpenCL CPU respectable**: 3.7x speedup, but beaten by OpenMP

## Why These Results?

**OpenMP Advantages:**
- Shared L3 cache (no data copying)
- 16 cores with efficient work distribution
- Minimal thread synchronization overhead

**GPU Disadvantages:**
- PCIe transfer overhead (67MB matrix + vectors)
- Memory-bound operation (ratio compute:memory too low)
- Can't compensate with raw parallelism

## Hardware Info

- CPU: Intel i7-11850H (16 threads via OpenMP)
- OpenMP version: 2.0 (MSVC limitation)
- C++ std parallelism: Uses thread pool

## Building

```cmd
build.bat
```

## Compiler Note

MSVC OpenMP 2.0 doesn't support `collapse(2)` clause, so only outer loop is parallelized. This limits OpenMP's potential performance.

## Next Steps

See `006_matrix_multiply` where O(n³) computation finally lets GPUs demonstrate superiority.