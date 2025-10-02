# 004: Asynchronous Multi-Device Execution

Demonstrates distributing work across multiple OpenCL devices simultaneously.

## Purpose

Show how to:
- Create separate contexts for devices from different platforms
- Launch kernels concurrently across all devices
- Combine results from heterogeneous hardware

## What It Does

1. Enumerates all OpenCL devices (NVIDIA GPU, Intel GPU, Intel CPU)
2. Creates individual context and command queue for each device
3. Splits a 16M element vector addition across devices
4. Launches all kernels simultaneously
5. Measures individual device execution times

## Architecture

**Key Limitation**: Cannot create a single `cl_context` with devices from multiple platforms. Each platform requires its own context.

```
Platform 0 (NVIDIA) → Context 0 → Device 0 (RTX A2000)
Platform 1 (Intel)  → Context 1 → Device 1 (UHD Graphics)
Platform 2 (Intel)  → Context 2 → Device 2 (i7 CPU)
```

Each device processes approximately 5.6M elements (16M / 3).

## Results

```
Total array size: 16777216 elements
Chunk size per device: 5592405 elements

Device 0 (NVIDIA RTX A2000):  Duration: 0.400 ms
Device 1 (Intel UHD Graphics): Duration: 2.071 ms
Device 2 (Intel i7-11850H):    Duration: 3.923 ms
```

## Important Note: Timing Analysis Limitation

The "Concurrency Analysis" showing "Sequential" execution is **misleading**. 

**Why**: Each OpenCL platform uses its own time reference. NVIDIA's timestamps use a different clock than Intel's, making direct comparison impossible. The devices likely **did** execute concurrently, but profiling events cannot prove it across vendors.

**Correct Approach**: Use host-side timing (`std::chrono`) to measure total wall-clock time for true concurrency verification.

## Building

```cmd
build.bat
```

## Key Concepts

- **Multiple contexts**: Required for multi-vendor execution
- **Command queues**: One per device for asynchronous submission
- **Profiling events**: Only comparable within same platform

## Lessons Learned

1. OpenCL contexts are platform-specific
2. Cross-platform timing requires host-side measurement
3. Multi-device execution works but is complex to verify