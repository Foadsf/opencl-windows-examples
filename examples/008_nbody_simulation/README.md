# 008: N-Body Simulation - O(n²) Gravitational Forces

Demonstrates GPU performance on O(n²) all-pairs force calculations, showing clear OpenCL advantage at scale.

## Purpose

Show **when OpenCL becomes optimal** for computationally intensive O(n²) algorithms by testing gravitational N-body simulation across particle counts from 128 to 4096.

## What It Does

Simulates gravitational forces between all particle pairs using:
1. **Serial C++**: Baseline nested loop
2. **OpenMP**: Multi-threaded CPU parallelization  
3. **OpenCL (simple)**: Direct GPU implementation
4. **OpenCL (tiled)**: Local memory optimization

Each particle computes forces from all other particles: O(n²) complexity.

## Key Results

### Breakeven Analysis

```
Particles    Serial    OpenMP    Best OpenCL    Winner
--------------------------------------------------------
128          0.04ms    2.16ms    0.09ms         Serial (overhead)
256          0.26ms    0.04ms    0.07ms         OpenMP (6x)
512          1.06ms    0.19ms    0.11ms         OpenCL (10x)
1024         4.15ms    0.38ms    0.15ms         OpenCL (28x)
2048        16.59ms    1.55ms    0.38ms         OpenCL (44x)
4096        66.88ms    7.02ms    0.95ms         OpenCL (70x)
```

**Critical Threshold: ~512 particles** - below this, overhead dominates; above this, OpenCL provides exponentially growing advantage.

### Performance at 4096 Particles

```
Implementation            Time        Speedup
------------------------------------------------
Serial C++               66.88ms      1.00x
OpenMP                    7.02ms      9.52x
NVIDIA GPU (tiled)        0.95ms     70.19x ⭐
Intel CPU OpenCL          1.32ms     50.55x
Intel UHD GPU             4.39ms     15.22x
```

NVIDIA GPU achieves **70x speedup** - the most dramatic GPU advantage we've seen yet.

## Why OpenCL Wins Here

**N-body has optimal GPU characteristics:**

1. **High compute-to-memory ratio**: n² operations per timestep, only n memory reads
2. **Perfect parallelism**: Each particle computed independently
3. **Regular computation**: No branching, same calculation repeated
4. **Scales with problem size**: Larger n → better GPU utilization

### Complexity Analysis

At 4096 particles:
- Force calculations: **16.7 million** (n × (n-1))
- Memory reads: 16KB positions + 16KB masses
- Operations-per-byte: **~520:1** ratio

This high arithmetic intensity is exactly what GPUs excel at.

## Surprising Discovery: Intel CPU OpenCL Dominance

Intel CPU OpenCL outperforms OpenMP and even GPUs at mid-range scales:

```
2048 particles:
OpenMP:                1.55ms (11x speedup)
Intel CPU OpenCL:      0.38ms (44x speedup) ⭐
NVIDIA GPU:            0.93ms (18x speedup)
```

**Why Intel CPU wins:**
1. **AVX2/AVX-512 SIMD**: 8-16 operations per instruction
2. **No PCIe overhead**: Data never leaves CPU memory
3. **Cache efficiency**: 2048 particles (~32KB) fits in L1/L2
4. **Better than OpenMP**: Superior compiler optimization and thread scheduling

This proves **OpenCL on CPU is a legitimate optimization strategy**, not just for testing.

## Optimization: Local Memory Tiling

Local memory caching provides modest improvements:

```
NVIDIA GPU at 4096:
Simple:  1.04ms
Tiled:   0.95ms (9% faster)
```

Unlike image convolution, N-body has irregular access patterns (all-to-all forces), limiting tiling effectiveness. The inner loop reads all n particles regardless, so caching provides marginal benefit.

## Performance Scaling

GPU advantage grows exponentially with particle count:

```
NVIDIA GPU Speedup:
128 particles:   0.1x (overhead dominates)
512 particles:   2.0x (break-even)
2048 particles:  18x
4096 particles:  70x
```

Doubling particles quadruples work (O(n²)), but GPU overhead remains constant, improving efficiency.

## When OpenCL Becomes Optimal

| Particle Count | Best Technology | Why |
|----------------|-----------------|-----|
| < 256 | Serial/OpenMP | Overhead too high |
| 256-512 | OpenMP | Quick 6x speedup |
| 512-2048 | Intel CPU OpenCL | 10-44x, no transfer overhead |
| 2048+ | GPU OpenCL | 70x+, massive parallelism |

**Rule of thumb**: Use OpenCL when n > 512 for O(n²) algorithms.

## Real-World Applications

This pattern applies to:
- Gravitational simulations (astrophysics, galaxy formation)
- Molecular dynamics (protein folding, drug design)
- Electrostatics (particle physics, plasma simulation)
- Collision detection (game engines, robotics)
- All-pairs algorithms (clustering, nearest-neighbor search)

Any O(n²) all-pairs calculation benefits from GPU acceleration at scale.

## Building

```cmd
build.bat
```

## Comparison with Other Examples

**Why N-body succeeds where others failed:**

| Example | Complexity | Ops/Memory | GPU Speedup |
|---------|-----------|------------|-------------|
| Vector addition | O(n) | 1 | 0.25x (loses) |
| Matrix-vector | O(n²) | ~10 | 4x (marginal) |
| Matrix multiply | O(n³) | ~2000 | 155x (wins) |
| Convolution (small) | O(n²) | 9 | 3x (marginal) |
| Convolution (large) | O(n²) | 225 | 150x (wins) |
| **N-body** | **O(n²)** | **~500** | **70x (wins)** |

Arithmetic intensity (operations per memory access) determines GPU advantage more than algorithmic complexity.

## Key Takeaways

1. **Clear threshold exists**: OpenCL becomes optimal above ~512 particles
2. **Exponential scaling**: GPU advantage grows quadratically with problem size
3. **CPU OpenCL is viable**: Can outperform discrete GPUs for mid-range problems
4. **OpenMP plateaus**: Limited to 10-12x speedup regardless of problem size
5. **Tiling has limits**: Local memory optimization less effective for irregular access

## Next Steps

To improve N-body further:
- Implement Barnes-Hut algorithm (O(n log n))
- Use tree-based force approximation
- Optimize for specific GPU architectures
- Implement GPU-GPU communication for multi-GPU scaling
```

Commit this example:

```cmd
cd C:\dev\OpenCL\20251002\opencl-windows-cpp
git add examples/008_nbody_simulation/
git commit -m "Add Example 008: N-Body Simulation

Demonstrates O(n²) gravitational forces across 128-4096 particles:
- Clear breakeven point at ~512 particles
- NVIDIA GPU achieves 70x speedup at 4096 particles
- Intel CPU OpenCL wins mid-range (44x at 2048 particles)
- OpenMP plateaus at 10-12x regardless of scale
- Proves OpenCL on CPU is a legitimate optimization strategy

Key insight: Arithmetic intensity matters more than complexity class"