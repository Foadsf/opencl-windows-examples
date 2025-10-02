#define CL_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <cstring>

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

// Serial C++ implementation
void vectorAddCPU(const std::vector<float>& a, 
                  const std::vector<float>& b, 
                  std::vector<float>& result) {
    for (size_t i = 0; i < a.size(); i++) {
        result[i] = a[i] + b[i];
    }
}

// OpenCL implementation
double vectorAddOpenCL(const std::vector<float>& a,
                       const std::vector<float>& b,
                       std::vector<float>& result,
                       cl_device_id device,
                       const char* deviceName) {
    
    cl_int err;
    size_t n = a.size();

    // Create context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");

    // Create command queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    checkError(err, "clCreateCommandQueue");

    // Create buffers
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
                                     n * sizeof(float), (void*)a.data(), &err);
    checkError(err, "clCreateBuffer A");

    cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     n * sizeof(float), (void*)b.data(), &err);
    checkError(err, "clCreateBuffer B");

    cl_mem bufferResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                          n * sizeof(float), nullptr, &err);
    checkError(err, "clCreateBuffer Result");

    // Load and build kernel
    std::string kernelSource = loadKernelSource("vector_add.cl");
    const char* kernelSourcePtr = kernelSource.c_str();
    size_t kernelSourceSize = kernelSource.size();

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourcePtr, 
                                                     &kernelSourceSize, &err);
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build error:\n" << log.data() << "\n";
        exit(1);
    }

    cl_kernel kernel = clCreateKernel(program, "vector_add", &err);
    checkError(err, "clCreateKernel");

    // Set kernel arguments
    unsigned int nArg = (unsigned int)n;
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufferB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufferResult);
    clSetKernelArg(kernel, 3, sizeof(unsigned int), &nArg);

    // Execute kernel and measure time
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t globalWorkSize = n;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, 
                                  nullptr, 0, nullptr, nullptr);
    checkError(err, "clEnqueueNDRangeKernel");

    clFinish(queue);
    
    auto end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double, std::milli>(end - start).count();

    // Read result
    err = clEnqueueReadBuffer(queue, bufferResult, CL_TRUE, 0, 
                               n * sizeof(float), result.data(), 0, nullptr, nullptr);
    checkError(err, "clEnqueueReadBuffer");

    // Cleanup
    clReleaseMemObject(bufferA);
    clReleaseMemObject(bufferB);
    clReleaseMemObject(bufferResult);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return elapsed;
}

void verifyResults(const std::vector<float>& expected, 
                   const std::vector<float>& actual,
                   const char* name) {
    bool correct = true;
    for (size_t i = 0; i < expected.size(); i++) {
        if (std::abs(expected[i] - actual[i]) > 0.001f) {
            std::cerr << "Mismatch in " << name << " at index " << i 
                      << ": expected " << expected[i] 
                      << ", got " << actual[i] << "\n";
            correct = false;
            break;
        }
    }
    if (correct) {
        std::cout << "  ✓ Results verified correct\n";
    }
}

int main() {
    std::cout << "=== Vector Addition Performance Comparison ===\n\n";

    // Problem size
    const size_t N = 10000000;  // 10 million elements
    std::cout << "Vector size: " << N << " elements\n";
    std::cout << "Memory per vector: " << (N * sizeof(float) / (1024.0 * 1024.0)) << " MB\n\n";

    // Initialize vectors
    std::vector<float> a(N), b(N);
    for (size_t i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    // Get all platforms and devices
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    std::vector<cl_device_id> allDevices;
    std::vector<std::string> deviceNames;

    for (cl_uint i = 0; i < numPlatforms; i++) {
        cl_uint numDevices;
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        std::vector<cl_device_id> devices(numDevices);
        clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);

        for (cl_uint j = 0; j < numDevices; j++) {
            char name[128];
            clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(name), name, nullptr);
            allDevices.push_back(devices[j]);
            deviceNames.push_back(std::string(name));
        }
    }

    // 1. Serial C++ benchmark
    std::cout << "===================================\n";
    std::cout << "1. Serial C++ (Single-threaded CPU)\n";
    std::cout << "===================================\n";
    
    std::vector<float> resultCPU(N);
    auto start = std::chrono::high_resolution_clock::now();
    vectorAddCPU(a, b, resultCPU);
    auto end = std::chrono::high_resolution_clock::now();
    double cpuTime = std::chrono::duration<double, std::milli>(end - start).count();
    
    std::cout << "Time: " << std::fixed << std::setprecision(2) << cpuTime << " ms\n";
    std::cout << "Speedup: 1.00x (baseline)\n\n";

    // 2. OpenCL on all devices
    for (size_t i = 0; i < allDevices.size(); i++) {
        std::cout << "===================================\n";
        std::cout << (i + 2) << ". OpenCL: " << deviceNames[i] << "\n";
        std::cout << "===================================\n";

        std::vector<float> resultOpenCL(N);
        double openclTime = vectorAddOpenCL(a, b, resultOpenCL, allDevices[i], deviceNames[i].c_str());

        std::cout << "Time: " << std::fixed << std::setprecision(2) << openclTime << " ms\n";
        std::cout << "Speedup: " << std::fixed << std::setprecision(2) << (cpuTime / openclTime) << "x\n";
        
        verifyResults(resultCPU, resultOpenCL, deviceNames[i].c_str());
        std::cout << "\n";
    }

    // Summary table
    std::cout << "===================================\n";
    std::cout << "Summary\n";
    std::cout << "===================================\n";
    std::cout << std::left << std::setw(40) << "Device" 
              << std::right << std::setw(12) << "Time (ms)" 
              << std::setw(12) << "Speedup\n";
    std::cout << std::string(64, '-') << "\n";
    std::cout << std::left << std::setw(40) << "Serial C++" 
              << std::right << std::setw(12) << std::fixed << std::setprecision(2) << cpuTime
              << std::setw(12) << "1.00x\n";

    return 0;
}