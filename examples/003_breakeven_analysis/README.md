# 003: Breakeven Analysis

Systematically tests vector sizes from 1K to 128M elements to find where OpenCL becomes faster than serial C++.

## Purpose

Identify the **minimum problem size** where GPU acceleration pays off for each device.

## What It Does

Tests vector addition at exponentially increasing sizes (1K, 4K, 16K, 64K, 256K, 1M, 4M, 16M, 64M, 128M) and measures:
- Serial C++ execution time (baseline)
- OpenCL execution time for each device
- Identifies breakeven point where OpenCL < Serial

Runs 5 iterations per size and reports the best time.

## Results Summary

```
Breakeven Points (where OpenCL becomes faster):

NVIDIA RTX A2000 Laptop GPU: 65536 elements (64K)
Intel(R) UHD Graphics:       262144 elements (256K)
Intel CPU (OpenCL):          65536 elements (64K)
```

## Performance Timeline

| Size | Serial C++ | NVIDIA GPU | Intel UHD | Intel CPU |
|------|-----------|------------|-----------|-----------|
| 1K   | 0.000ms   | 0.029ms    | 0.052ms   | 0.017ms   |
| 64K  | 0.038ms   | **0.031ms** ✓ | 0.061ms   | **0.038ms** ✓ |
| 256K | 0.105ms   | 0.043ms    | **0.086ms** ✓ | 0.085ms   |
| 64M  | 39.521ms  | **4.835ms** | **19.953ms** | **48.331ms** |

## Key Insights

1. **Small data (< 64K)**: CPU always wins due to cache efficiency
2. **Medium data (64K-1M)**: GPUs start showing benefit
3. **Large data (> 1M)**: GPUs provide significant speedup

**But**: Even at 128M elements, speedup is only 8-10x because vector addition remains memory-bound.

## Building

```cmd
build.bat
```

## Next Steps

See `006_matrix_multiply` for operations where GPUs provide 100x+ speedup.