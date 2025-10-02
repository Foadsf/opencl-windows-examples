#define CL_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <string>
#include <sstream>

std::string loadKernelSource(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open: " << filename << "\n";
        return "";
    }
    return std::string(std::istreambuf_iterator<char>(file),
                       std::istreambuf_iterator<char>());
}

void checkError(cl_int err, const char* operation) {
    if (err != CL_SUCCESS) {
        std::cerr << "Error during " << operation << ": " << err << "\n";
        exit(1);
    }
}

double vectorAddCPU(const std::vector<float>& a, 
                    const std::vector<float>& b, 
                    std::vector<float>& result,
                    int iterations = 5) {
    double minTime = 1e9;
    
    for (int iter = 0; iter < iterations; iter++) {
        auto start = std::chrono::high_resolution_clock::now();
        for (size_t i = 0; i < a.size(); i++) {
            result[i] = a[i] + b[i];
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        minTime = std::min(minTime, elapsed);
    }
    
    return minTime;
}

double vectorAddOpenCL(const std::vector<float>& a,
                       const std::vector<float>& b,
                       std::vector<float>& result,
                       cl_device_id device,
                       cl_context context,
                       cl_program program,
                       int iterations = 5) {
    
    cl_int err;
    size_t n = a.size();
    double minTime = 1e9;

    // Create command queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    checkError(err, "clCreateCommandQueue");

    // Create buffers
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY, 
                                     n * sizeof(float), nullptr, &err);
    checkError(err, "clCreateBuffer A");

    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY,
                                     n * sizeof(float), nullptr, &err);
    checkError(err, "clCreateBuffer B");

    cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                          n * sizeof(float), nullptr, &err);
    checkError(err, "clCreateBuffer Result");

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    checkError(err, "clCreateKernel");

    // Set kernel arguments
    unsigned int nArg = (unsigned int)n;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &nArg);

    for (int iter = 0; iter < iterations; iter++) {
        // Write buffers
        clEnqueueWriteBuffer(queue, bufferA, CL_TRUE, 0, n * sizeof(float), a.data(), 0, nullptr, nullptr);
        clEnqueueWriteBuffer(queue, bufferB, CL_TRUE, 0, n * sizeof(float), b.data(), 0, nullptr, nullptr);

        // Execute kernel and measure time
        auto start = std::chrono::high_resolution_clock::now();
        
        size_t globalWorkSize = n;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, 
                                      nullptr, 0, nullptr, nullptr);
        checkError(err, "clEnqueueNDRangeKernel");
        clFinish(queue);
        
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        minTime = std::min(minTime, elapsed);

        // Read result
        if (iter == 0) {  // Only read back on first iteration for verification
            clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, 
                               n * sizeof(float), result.data(), 0, nullptr, nullptr);
        }
    }

    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);

    return minTime;
}

struct DeviceInfo {
    cl_device_id id;
    std::string name;
    cl_device_type type;
};

int main() {
    std::cout << "=== OpenCL Breakeven Point Analysis ===\n\n";
    std::cout << "Finding the vector size where OpenCL becomes faster than serial C++\n\n";

    // Get all OpenCL devices
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    std::vector<DeviceInfo> devices;
    for (cl_uint i = 0; i < numPlatforms; i++) {
        cl_uint numDevices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        std::vector<cl_device_id> platformDevices(numDevices);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, platformDevices.data(), nullptr);

        for (cl_uint j = 0; j < numDevices; j++) {
            DeviceInfo info;
            info.id = platformDevices[j];
            
            char name[128];
            clGetDeviceInfo(info.id, CL_DEVICE_NAME, sizeof(name), name, nullptr);
            info.name = name;
            
            clGetDeviceInfo(info.id, CL_DEVICE_TYPE, sizeof(info.type), &info.type, nullptr);
            
            devices.push_back(info);
        }
    }

    std::cout << "Testing on " << devices.size() << " OpenCL device(s):\n";
    for (size_t i = 0; i < devices.size(); i++) {
        std::cout << "  " << (i + 1) << ". " << devices[i].name;
        if (devices[i].type & CL_DEVICE_TYPE_GPU) std::cout << " (GPU)";
        if (devices[i].type & CL_DEVICE_TYPE_CPU) std::cout << " (CPU)";
        std::cout << "\n";
    }
    std::cout << "\n";

    // Prepare OpenCL contexts and programs for each device
    std::vector<cl_context> contexts;
    std::vector<cl_program> programs;
    
    std::string kernelSource = loadKernelSource("vector_add.cl");
    const char* kernelSourcePtr = kernelSource.c_str();
    size_t kernelSourceSize = kernelSource.size();

    for (const auto& device : devices) {
        cl_int err;
        cl_context context = clCreateContext(nullptr, 1, &device.id, nullptr, nullptr, &err);
        checkError(err, "clCreateContext");
        contexts.push_back(context);

        cl_program program = clCreateProgramWithSource(context, 1, &kernelSourcePtr, &kernelSourceSize, &err);
        checkError(err, "clCreateProgramWithSource");
        
        err = clBuildProgram(program, 1, &device.id, nullptr, nullptr, nullptr);
        if (err != CL_SUCCESS) {
            size_t logSize;
            clGetProgramBuildInfo(program, device.id, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::vector<char> log(logSize);
            clGetProgramBuildInfo(program, device.id, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
            std::cerr << "Build error for " << device.name << ":\n" << log.data() << "\n";
            exit(1);
        }
        programs.push_back(program);
    }

    // Test different vector sizes (powers of 2)
    std::vector<size_t> sizes = {
        1024,           // 1K
        4096,           // 4K
        16384,          // 16K
        65536,          // 64K
        262144,         // 256K
        1048576,        // 1M
        4194304,        // 4M
        16777216,       // 16M
        67108864,       // 64M
        134217728       // 128M
    };

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Running tests (best of 5 iterations per size)...\n\n";

    // Table header
    std::cout << std::left << std::setw(12) << "Size"
              << std::right << std::setw(12) << "Elements"
              << std::setw(12) << "CPU (ms)";
    
    for (const auto& device : devices) {
        std::string shortName = device.name.substr(0, 10);
        std::cout << std::setw(12) << shortName;
    }
    std::cout << "\n" << std::string(12 + 12 + 12 + devices.size() * 12, '-') << "\n";

    // Track breakeven points
    std::vector<size_t> breakevenPoints(devices.size(), 0);
    std::vector<bool> foundBreakeven(devices.size(), false);

    for (size_t testSize : sizes) {
        // Initialize test vectors
        std::vector<float> a(testSize), b(testSize);
        for (size_t i = 0; i < testSize; i++) {
            a[i] = static_cast<float>(i % 1000);
            b[i] = static_cast<float>((i * 2) % 1000);
        }

        // CPU baseline
        std::vector<float> resultCPU(testSize);
        double cpuTime = vectorAddCPU(a, b, resultCPU);

        // Size label
        std::string sizeLabel;
        std::ostringstream oss;
        if (testSize >= 1048576) {
            oss << (testSize / 1048576) << "M";
            sizeLabel = oss.str();
        } else if (testSize >= 1024) {
            oss << (testSize / 1024) << "K";
            sizeLabel = oss.str();
        } else {
            oss << testSize;
            sizeLabel = oss.str();
        }

        std::cout << std::left << std::setw(12) << sizeLabel
                  << std::right << std::setw(12) << testSize
                  << std::setw(12) << cpuTime;

        // Test each OpenCL device
        for (size_t i = 0; i < devices.size(); i++) {
            std::vector<float> resultOpenCL(testSize);
            double openclTime = vectorAddOpenCL(a, b, resultOpenCL, devices[i].id, 
                                                 contexts[i], programs[i]);
            
            std::cout << std::setw(12) << openclTime;

            // Check for breakeven point
            if (!foundBreakeven[i] && openclTime < cpuTime) {
                breakevenPoints[i] = testSize;
                foundBreakeven[i] = true;
            }
        }
        std::cout << "\n";
    }

    // Summary
    std::cout << "\n=== Breakeven Points (where OpenCL becomes faster) ===\n\n";
    for (size_t i = 0; i < devices.size(); i++) {
        std::cout << devices[i].name << ": ";
        if (foundBreakeven[i]) {
            std::cout << breakevenPoints[i] << " elements\n";
        } else {
            std::cout << "Not reached (OpenCL slower for all tested sizes)\n";
        }
    }

    // Cleanup
    for (auto& program : programs) clReleaseProgram(program);
    for (auto& context : contexts) clReleaseContext(context);

    return 0;
}