# 000: Device Enumeration

Detects and displays all OpenCL platforms and devices available on your system.

## Purpose

- Learn how to query OpenCL platforms
- Discover available computing devices (CPUs, GPUs)
- Understand device capabilities and properties

## What It Does

1. Enumerates all OpenCL platforms (NVIDIA, Intel, AMD, etc.)
2. Lists devices under each platform
3. Displays device properties:
   - Device name
   - Type (GPU/CPU/Accelerator)
   - Global memory size
   - Compute units (cores/execution units)

## Expected Output

```
=== OpenCL Device Enumeration ===

Found 3 OpenCL platform(s)

Platform 0:
  Name: NVIDIA CUDA
  Vendor: NVIDIA Corporation
  Version: OpenCL 3.0 CUDA 12.9.40
  Attempting device enumeration...
  Devices: 1
    Device 0:
      Name: NVIDIA RTX A2000 Laptop GPU
      Type: GPU
      Global Memory: 4095 MB
      Compute Units: 20

Platform 1:
  Name: Intel(R) OpenCL Graphics
  Vendor: Intel(R) Corporation
  Version: OpenCL 3.0
  Attempting device enumeration...
  Devices: 1
    Device 0:
      Name: Intel(R) UHD Graphics
      Type: GPU
      Global Memory: 14792 MB
      Compute Units: 32

Platform 2:
  Name: Intel(R) OpenCL
  Vendor: Intel(R) Corporation
  Version: OpenCL 3.0 WINDOWS
  Attempting device enumeration...
  Devices: 1
    Device 0:
      Name: 11th Gen Intel(R) Core(TM) i7-11850H @ 2.50GHz
      Type: CPU
      Global Memory: 32431 MB
      Compute Units: 16

Enumeration complete!
```

## Building

```cmd
build.bat
```

## Key Concepts

- **Platform**: OpenCL vendor implementation (NVIDIA, Intel, AMD)
- **Device**: Physical computing hardware (GPU, CPU)
- **Compute Units**: Parallel processing cores
- **Global Memory**: Device RAM available for computations

## Troubleshooting

**No platforms found**: Install at least one OpenCL runtime (NVIDIA CUDA, Intel Graphics Driver, or Intel oneAPI)

**Program hangs**: See LESSONS_LEARNED.md - likely caused by conflicting Intel OpenCL runtimes

## Next Steps

See `001_hello_opencl` for your first kernel execution.