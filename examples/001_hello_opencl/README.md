# 001 - Hello OpenCL

The simplest possible OpenCL program: a kernel that writes "Hello from GPU!" to a buffer.

## What This Example Demonstrates

- Loading an OpenCL kernel from a `.cl` file
- Creating a context and command queue
- Compiling a kernel at runtime
- Creating a buffer and passing it to a kernel
- Reading results back from the GPU

## Building

```cmd
build.bat
```

## Expected Output

```
=== OpenCL Hello World ===

Using platform: NVIDIA CUDA
Using device: NVIDIA RTX A2000 Laptop GPU

Executing kernel...
Kernel output: Hello from GPU!

Success!
```

## Key Concepts

- **Kernel**: The `.cl` file contains code that runs on the GPU
- **Work-item**: Each thread executing the kernel (we use only 1 here)
- **Buffer**: Memory allocated on the GPU (`cl_mem`)
- **Command Queue**: Where we submit operations to execute

## Next Steps

See `002_vector_addition` for a more practical example with performance comparison.
