#define CL_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <execution>
#include <numeric>
#include <cmath>
#include <omp.h>

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

// 1. Serial implementation
double matmulSerial(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int M, int N, int K) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// 2. C++17 std::execution::par
double matmulStdPar(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int M, int N, int K) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> rows(M);
    std::iota(rows.begin(), rows.end(), 0);
    
    std::for_each(std::execution::par, rows.begin(), rows.end(),
        [&](int i) {
            for (int j = 0; j < K; j++) {
                float sum = 0.0f;
                for (int k = 0; k < N; k++) {
                    sum += A[i * N + k] * B[k * K + j];
                }
                C[i * K + j] = sum;
            }
        });
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// 3. OpenMP
double matmulOpenMP(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int M, int N, int K) {
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < K; j++) {
            float sum = 0.0f;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// 4. OpenCL (simple version)
double matmulOpenCL(const std::vector<float>& A,
                    const std::vector<float>& B,
                    std::vector<float>& C,
                    int M, int N, int K,
                    cl_device_id device,
                    cl_context context,
                    cl_program program,
                    bool useTiled = false) {
    cl_int err;
    
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    checkError(err, "clCreateCommandQueue");
    
    size_t bufSizeA = M * N * sizeof(float);
    size_t bufSizeB = N * K * sizeof(float);
    size_t bufSizeC = M * K * sizeof(float);
    
    cl_mem bufA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  bufSizeA, (void*)A.data(), &err);
    checkError(err, "clCreateBuffer A");
    
    cl_mem bufB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                  bufSizeB, (void*)B.data(), &err);
    checkError(err, "clCreateBuffer B");
    
    cl_mem bufC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, bufSizeC, nullptr, &err);
    checkError(err, "clCreateBuffer C");
    
    const char* kernelName = useTiled ? "matrix_multiply_tiled" : "matrix_multiply";
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    checkError(err, "clCreateKernel");
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufA);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufB);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufC);
    clSetKernelArg(kernel, 3, sizeof(int), &M);
    clSetKernelArg(kernel, 4, sizeof(int), &N);
    clSetKernelArg(kernel, 5, sizeof(int), &K);
    
    if (useTiled) {
        const int TILE_SIZE = 16;
        size_t localMemSize = TILE_SIZE * TILE_SIZE * sizeof(float);
        clSetKernelArg(kernel, 6, localMemSize, nullptr);
        clSetKernelArg(kernel, 7, localMemSize, nullptr);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (useTiled) {
        const int TILE_SIZE = 16;
        size_t globalSize[2] = {(size_t)((M + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE,
                                (size_t)((K + TILE_SIZE - 1) / TILE_SIZE) * TILE_SIZE};
        size_t localSize[2] = {TILE_SIZE, TILE_SIZE};
        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, localSize, 0, nullptr, nullptr);
    } else {
        size_t globalSize[2] = {(size_t)M, (size_t)K};
        err = clEnqueueNDRangeKernel(queue, kernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    }
    checkError(err, "clEnqueueNDRangeKernel");
    
    clFinish(queue);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    clEnqueueReadBuffer(queue, bufC, CL_TRUE, 0, bufSizeC, C.data(), 0, nullptr, nullptr);
    
    clReleaseMemObject(bufA);
    clReleaseMemObject(bufB);
    clReleaseMemObject(bufC);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

void verifyResults(const std::vector<float>& expected, const std::vector<float>& actual, const char* name) {
    const float EPSILON = 0.01f;
    bool correct = true;
    int errors = 0;
    
    for (size_t i = 0; i < expected.size(); i++) {
        if (std::abs(expected[i] - actual[i]) > EPSILON) {
            if (errors < 5) {
                std::cerr << "Mismatch in " << name << " at " << i 
                          << ": expected " << expected[i] 
                          << ", got " << actual[i] << "\n";
            }
            errors++;
            correct = false;
        }
    }
    
    if (correct) {
        std::cout << "  ✓ Verified\n";
    } else {
        std::cout << "  ✗ Failed (" << errors << " errors)\n";
    }
}

int main() {
    std::cout << "=== Matrix Multiplication Performance Comparison ===\n\n";
    
    // Test sizes - square matrices
    std::vector<int> sizes = {256, 512, 1024, 2048};
    
    // Get OpenCL devices
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    
    std::vector<cl_device_id> devices;
    std::vector<std::string> deviceNames;
    std::vector<cl_context> contexts;
    std::vector<cl_program> programs;
    
    std::string kernelSource = loadKernelSource("matmul.cl");
    const char* kernelSourcePtr = kernelSource.c_str();
    size_t kernelSourceSize = kernelSource.size();
    
    for (cl_uint p = 0; p < numPlatforms; p++) {
        cl_uint numDevices;
        cl_int err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        if (err == CL_SUCCESS && numDevices > 0) {
            std::vector<cl_device_id> platformDevices(numDevices);
            clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, numDevices, platformDevices.data(), nullptr);
            
            for (cl_uint d = 0; d < numDevices; d++) {
                devices.push_back(platformDevices[d]);
                
                char name[128];
                clGetDeviceInfo(platformDevices[d], CL_DEVICE_NAME, sizeof(name), name, nullptr);
                deviceNames.push_back(std::string(name));
                
                cl_context context = clCreateContext(nullptr, 1, &platformDevices[d], nullptr, nullptr, &err);
                contexts.push_back(context);
                
                cl_program program = clCreateProgramWithSource(context, 1, &kernelSourcePtr, &kernelSourceSize, &err);
                err = clBuildProgram(program, 1, &platformDevices[d], nullptr, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    size_t logSize;
                    clGetProgramBuildInfo(program, platformDevices[d], CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
                    std::vector<char> log(logSize);
                    clGetProgramBuildInfo(program, platformDevices[d], CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
                    std::cerr << "Build error for " << name << ":\n" << log.data() << "\n";
                }
                programs.push_back(program);
            }
        }
    }
    
    std::cout << "CPU Cores (OpenMP): " << omp_get_max_threads() << "\n";
    std::cout << "OpenCL Devices: " << devices.size() << "\n\n";
    
    for (int size : sizes) {
        int M = size, N = size, K = size;
        
        std::cout << "========================================\n";
        std::cout << "Matrix size: " << M << "x" << N << " × " << N << "x" << K << "\n";
        std::cout << "Operations: " << (2.0 * M * N * K / 1e9) << " GFLOP\n";
        std::cout << "========================================\n";
        
        // Initialize matrices
        std::vector<float> A(M * N);
        std::vector<float> B(N * K);
        std::vector<float> C(M * K);
        
        for (int i = 0; i < M * N; i++) A[i] = (float)(i % 100) / 100.0f;
        for (int i = 0; i < N * K; i++) B[i] = (float)(i % 100) / 100.0f;
        
        // 1. Serial
        std::cout << "\nSerial C++... ";
        std::cout.flush();
        double serialTime = matmulSerial(A, B, C, M, N, K);
        std::vector<float> expectedResult = C;
        std::cout << serialTime << " ms\n";
        
        // 2. C++ std::execution::par
        std::cout << "C++ std::execution::par... ";
        std::cout.flush();
        std::fill(C.begin(), C.end(), 0.0f);
        double stdParTime = matmulStdPar(A, B, C, M, N, K);
        std::cout << stdParTime << " ms\n";
        verifyResults(expectedResult, C, "std::par");
        
        // 3. OpenMP
        std::cout << "OpenMP... ";
        std::cout.flush();
        std::fill(C.begin(), C.end(), 0.0f);
        double openmpTime = matmulOpenMP(A, B, C, M, N, K);
        std::cout << openmpTime << " ms\n";
        verifyResults(expectedResult, C, "OpenMP");
        
        // Results table
        std::cout << "\n" << std::fixed << std::setprecision(2);
        std::cout << std::left << std::setw(35) << "Implementation"
                  << std::right << std::setw(12) << "Time (ms)"
                  << std::setw(12) << "GFLOPS"
                  << std::setw(12) << "Speedup\n";
        std::cout << std::string(71, '-') << "\n";
        
        double gflop = 2.0 * M * N * K / 1e9;
        
        std::cout << std::left << std::setw(35) << "Serial C++"
                  << std::right << std::setw(12) << serialTime
                  << std::setw(12) << (gflop / (serialTime / 1000.0))
                  << std::setw(12) << "1.00x\n";
        
        std::cout << std::left << std::setw(35) << "C++17 std::execution::par"
                  << std::right << std::setw(12) << stdParTime
                  << std::setw(12) << (gflop / (stdParTime / 1000.0))
                  << std::setw(12) << (serialTime / stdParTime) << "x\n";
        
        std::cout << std::left << std::setw(35) << "OpenMP"
                  << std::right << std::setw(12) << openmpTime
                  << std::setw(12) << (gflop / (openmpTime / 1000.0))
                  << std::setw(12) << (serialTime / openmpTime) << "x\n";
        
        // OpenCL devices
        for (size_t i = 0; i < devices.size(); i++) {
            std::cout << "\n" << deviceNames[i] << " (simple)... ";
            std::cout.flush();
            std::fill(C.begin(), C.end(), 0.0f);
            double openclTime = matmulOpenCL(A, B, C, M, N, K, devices[i], contexts[i], programs[i], false);
            std::cout << openclTime << " ms\n";
            verifyResults(expectedResult, C, deviceNames[i].c_str());
            
            std::string name = "OpenCL: " + deviceNames[i].substr(0, 22);
            std::cout << std::left << std::setw(35) << name
                      << std::right << std::setw(12) << openclTime
                      << std::setw(12) << (gflop / (openclTime / 1000.0))
                      << std::setw(12) << (serialTime / openclTime) << "x\n";
            
            // Tiled version
            std::cout << deviceNames[i] << " (tiled)... ";
            std::cout.flush();
            std::fill(C.begin(), C.end(), 0.0f);
            double tiledTime = matmulOpenCL(A, B, C, M, N, K, devices[i], contexts[i], programs[i], true);
            std::cout << tiledTime << " ms\n";
            verifyResults(expectedResult, C, (deviceNames[i] + " tiled").c_str());
            
            std::string tiledName = "OpenCL: " + deviceNames[i].substr(0, 22) + " (tiled)";
            std::cout << std::left << std::setw(35) << tiledName
                      << std::right << std::setw(12) << tiledTime
                      << std::setw(12) << (gflop / (tiledTime / 1000.0))
                      << std::setw(12) << (serialTime / tiledTime) << "x\n";
        }
        
        std::cout << "\n";
    }
    
    // Cleanup
    for (auto& prog : programs) clReleaseProgram(prog);
    for (auto& ctx : contexts) clReleaseContext(ctx);
    
    return 0;
}