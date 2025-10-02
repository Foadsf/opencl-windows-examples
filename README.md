# OpenCL Examples for Windows (C/C++)

A collection of practical OpenCL examples demonstrating GPU computing on Windows using Visual Studio 2019 and CMake.

## System Requirements

- **OS**: Windows 10/11
- **Compiler**: Visual Studio 2019 Build Tools (MSVC 19.29+)
- **CMake**: 3.15+ (bundled with VS Build Tools)
- **OpenCL Runtimes**: At least one of:
  - NVIDIA CUDA Toolkit 12.x (for NVIDIA GPUs)
  - Intel Graphics Driver (for Intel integrated GPUs)
  - Intel oneAPI Base Toolkit (for CPU execution)

## Hardware Tested

- **GPU**: NVIDIA RTX A2000 Laptop GPU
- **iGPU**: Intel UHD Graphics (11th Gen)
- **CPU**: Intel Core i7-11850H @ 2.50GHz

## Repository Structure

```
opencl-windows-cpp/
├── examples/
│   ├── 000_device_enumeration/    # List all OpenCL platforms and devices
│   ├── 001_hello_opencl/          # Simple "Hello World" kernel
│   ├── 002_vector_addition/       # CPU vs GPU performance comparison
│   ├── 003_breakeven_analysis/    # Find OpenCL performance crossover points
│   └── 004_async_multidevice/     # Concurrent execution across devices
├── setup/
│   ├── check_opencl_installed.bat # Verify OpenCL installation
│   └── detect_opencl_hardware.bat # Hardware detection script
└── README.md
```

## Quick Start

### 1. Verify OpenCL Installation

```cmd
cd setup
check_opencl_installed.bat
```

Expected output: Lists installed OpenCL runtimes and detects your hardware.

### 2. Build and Run an Example

```cmd
cd examples\001_hello_opencl
build.bat
```

Each example includes:
- `main.cpp` - Host code
- `*.cl` - OpenCL kernel(s)
- `CMakeLists.txt` - CMake configuration
- `build.bat` - Build and run script
- `README.md` - Example documentation

## Examples Overview

### 000: Device Enumeration
**Purpose**: Detect all OpenCL platforms and devices on your system.

**Key Concepts**: Platform querying, device properties

```cmd
cd examples\000_device_enumeration
build.bat
```

**Output**: Lists all GPUs and CPUs with their capabilities.

---

### 001: Hello OpenCL
**Purpose**: Simplest possible GPU kernel - write "Hello from GPU!" to a buffer.

**Key Concepts**: Context creation, kernel compilation, buffer management

```cmd
cd examples\001_hello_opencl
build.bat
```

**Expected Output**:
```
Using platform: NVIDIA CUDA
Using device: NVIDIA RTX A2000 Laptop GPU
Kernel output: Hello from GPU!
Success!
```

---

### 002: Vector Addition
**Purpose**: Compare CPU vs GPU performance for vector addition.

**Key Concepts**: Memory transfer overhead, parallel execution, performance measurement

```cmd
cd examples\002_vector_addition
build.bat
```

**Results** (10M elements):
- Serial C++: 6.12ms (baseline)
- NVIDIA GPU: 24.23ms (slower due to transfer overhead)
- Intel UHD GPU: 10.09ms
- Intel CPU: 7.72ms

**Lesson**: Simple operations are dominated by data transfer costs.

---

### 003: Breakeven Analysis
**Purpose**: Find the vector size where OpenCL becomes faster than serial C++.

**Key Concepts**: Performance profiling, scaling analysis, breakeven points

```cmd
cd examples\003_breakeven_analysis
build.bat
```

**Key Findings**:
- **NVIDIA RTX A2000**: Faster than CPU at 64K elements
- **Intel UHD Graphics**: Faster than CPU at 256K elements  
- **Intel CPU (OpenCL)**: Faster than serial at 65K elements

**Lesson**: GPUs require sufficient workload to amortize overhead.

---

### 004: Async Multi-Device
**Purpose**: Execute kernels simultaneously across multiple devices.

**Key Concepts**: Multiple contexts, command queues, asynchronous execution

```cmd
cd examples\004_async_multidevice
build.bat
```

**Important Note**: Each platform uses its own time reference, so timing comparisons across platforms are unreliable. Devices do execute concurrently, but profiling events can't directly prove it.

## Building from Source

All examples use the same build pattern:

```cmd
cd examples\<example_name>
build.bat
```

The `build.bat` script:
1. Creates a `build/` directory
2. Runs CMake to generate Visual Studio projects
3. Builds in Release configuration
4. Copies kernel files (`.cl`) to output directory
5. Runs the executable

### Manual Build (Advanced)

```cmd
mkdir build && cd build
cmake .. -G "Visual Studio 16 2019" -A x64
cmake --build . --config Release
cd Release
<executable>.exe
```

## Troubleshooting

### "No OpenCL platforms found"
- Install at least one OpenCL runtime (NVIDIA CUDA, Intel Graphics Driver, or Intel oneAPI)
- Run `setup\check_opencl_installed.bat` to verify

### "Failed to open kernel file"
- Ensure `.cl` files are in the same directory as the executable
- Check that `build.bat` copies kernel files correctly

### Device enumeration hangs
- Update Intel Graphics drivers to latest version
- Remove legacy Intel OpenCL CPU Runtime if installed alongside Intel oneAPI
- See: https://github.com/intel/compute-runtime

### CMake not found
- Install Visual Studio 2019 Build Tools with "Desktop development with C++"
- Or install standalone CMake from https://cmake.org

## Performance Tips

1. **Memory-bound operations** (like vector addition) don't benefit much from GPUs due to transfer overhead
2. **Compute-intensive operations** (matrix multiplication, image processing) show significant GPU speedup
3. **Breakeven point** varies by hardware - test with realistic data sizes
4. **Multi-device execution** works best when devices have independent work chunks

## Learning Path

Recommended order:
1. **000_device_enumeration** - Understand your hardware
2. **001_hello_opencl** - Learn basic OpenCL workflow
3. **002_vector_addition** - See why simple operations are slow
4. **003_breakeven_analysis** - Find when GPU acceleration helps
5. **004_async_multidevice** - Advanced: use all devices simultaneously

## Common Issues & Solutions

### Issue: OpenCL hangs during platform enumeration
**Cause**: Conflicting Intel OpenCL runtimes  
**Solution**: Uninstall legacy "Intel OpenCL CPU Runtime 16.x", keep only Intel oneAPI

### Issue: Build errors about `std::to_string`
**Cause**: Missing `<string>` header  
**Solution**: Add `#include <string>` at top of file

### Issue: "Cannot create context with devices from multiple platforms"
**Cause**: Trying to use devices from NVIDIA + Intel in single context  
**Solution**: Create separate contexts per platform (see example 004)

## Resources

- **OpenCL Programming Guide**: https://www.khronos.org/opencl/
- **NVIDIA OpenCL Best Practices**: https://docs.nvidia.com/cuda/opencl-best-practices-guide/
- **Intel OpenCL Documentation**: https://www.intel.com/content/www/us/en/developer/tools/opencl-sdk/overview.html

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style
- Examples build cleanly on Windows + MSVC 2019
- Include README.md for new examples
- Test on at least one GPU platform

## Acknowledgments

Examples developed and tested on Windows 10 with:
- Visual Studio 2019 Build Tools
- NVIDIA CUDA Toolkit 12.9
- Intel oneAPI 2025.1
