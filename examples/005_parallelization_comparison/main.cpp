#define CL_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <execution>
#include <numeric>
#include <omp.h>

// Utility functions
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
double matvecSerial(const std::vector<float>& matrix,
                    const std::vector<float>& vector,
                    std::vector<float>& result,
                    int rows, int cols) {
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// 2. C++17 std::execution::par
double matvecStdPar(const std::vector<float>& matrix,
                    const std::vector<float>& vector,
                    std::vector<float>& result,
                    int rows, int cols) {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> indices(rows);
    std::iota(indices.begin(), indices.end(), 0);
    
    std::for_each(std::execution::par, indices.begin(), indices.end(),
        [&](int i) {
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                sum += matrix[i * cols + j] * vector[j];
            }
            result[i] = sum;
        });
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// 3. OpenMP
double matvecOpenMP(const std::vector<float>& matrix,
                    const std::vector<float>& vector,
                    std::vector<float>& result,
                    int rows, int cols) {
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// 4. OpenCL
double matvecOpenCL(const std::vector<float>& matrix,
                    const std::vector<float>& vector,
                    std::vector<float>& result,
                    int rows, int cols,
                    cl_device_id device,
                    cl_context context,
                    cl_program program) {
    cl_int err;
    
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    checkError(err, "clCreateCommandQueue");
    
    cl_mem bufMatrix = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       matrix.size() * sizeof(float), (void*)matrix.data(), &err);
    checkError(err, "clCreateBuffer matrix");
    
    cl_mem bufVector = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       vector.size() * sizeof(float), (void*)vector.data(), &err);
    checkError(err, "clCreateBuffer vector");
    
    cl_mem bufResult = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                                       result.size() * sizeof(float), nullptr, &err);
    checkError(err, "clCreateBuffer result");
    
    cl_kernel kernel = clCreateKernel(program, "matvec_multiply", &err);
    checkError(err, "clCreateKernel");
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufMatrix);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufVector);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufResult);
    clSetKernelArg(kernel, 3, sizeof(int), &rows);
    clSetKernelArg(kernel, 4, sizeof(int), &cols);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    size_t globalSize = rows;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    checkError(err, "clEnqueueNDRangeKernel");
    
    clFinish(queue);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    clEnqueueReadBuffer(queue, bufResult, CL_TRUE, 0, result.size() * sizeof(float),
                        result.data(), 0, nullptr, nullptr);
    
    clReleaseMemObject(bufMatrix);
    clReleaseMemObject(bufVector);
    clReleaseMemObject(bufResult);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "=== Parallelization Comparison: Matrix-Vector Multiplication ===\n\n";
    
    // Problem sizes to test
    std::vector<int> sizes = {512, 1024, 2048, 4096};
    
    // Get OpenCL devices
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    
    std::vector<cl_device_id> devices;
    std::vector<std::string> deviceNames;
    std::vector<cl_context> contexts;
    std::vector<cl_program> programs;
    
    std::string kernelSource = loadKernelSource("matvec.cl");
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
                clBuildProgram(program, 1, &platformDevices[d], nullptr, nullptr, nullptr);
                programs.push_back(program);
            }
        }
    }
    
    std::cout << "OpenMP threads available: " << omp_get_max_threads() << "\n";
    std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << "\n";
    std::cout << "OpenCL devices: " << devices.size() << "\n\n";
    
    for (int size : sizes) {
        int rows = size;
        int cols = size;
        
        std::cout << "========================================\n";
        std::cout << "Matrix size: " << rows << "x" << cols << "\n";
        std::cout << "========================================\n";
        
        // Initialize data
        std::vector<float> matrix(rows * cols);
        std::vector<float> vector(cols);
        std::vector<float> result(rows);
        
        for (int i = 0; i < rows * cols; i++) matrix[i] = static_cast<float>(i % 100) / 100.0f;
        for (int i = 0; i < cols; i++) vector[i] = static_cast<float>(i % 50) / 50.0f;
        
        // 1. Serial
        double serialTime = matvecSerial(matrix, vector, result, rows, cols);
        std::vector<float> expectedResult = result;
        
        // 2. C++ std::execution::par
        std::fill(result.begin(), result.end(), 0.0f);
        double stdParTime = matvecStdPar(matrix, vector, result, rows, cols);
        
        // 3. OpenMP
        std::fill(result.begin(), result.end(), 0.0f);
        double openmpTime = matvecOpenMP(matrix, vector, result, rows, cols);
        
        // Results table
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "\nImplementation                Time (ms)    Speedup\n";
        std::cout << "------------------------------------------------\n";
        std::cout << std::left << std::setw(28) << "Serial C++" 
                  << std::right << std::setw(10) << serialTime 
                  << std::setw(10) << "1.00x\n";
        std::cout << std::left << std::setw(28) << "C++17 std::execution::par" 
                  << std::right << std::setw(10) << stdParTime 
                  << std::setw(10) << (serialTime / stdParTime) << "x\n";
        std::cout << std::left << std::setw(28) << "OpenMP" 
                  << std::right << std::setw(10) << openmpTime 
                  << std::setw(10) << (serialTime / openmpTime) << "x\n";
        
        // 4. OpenCL devices
        for (size_t i = 0; i < devices.size(); i++) {
            std::fill(result.begin(), result.end(), 0.0f);
            double openclTime = matvecOpenCL(matrix, vector, result, rows, cols,
                                              devices[i], contexts[i], programs[i]);
            
            std::string name = "OpenCL: " + deviceNames[i].substr(0, 18);
            std::cout << std::left << std::setw(28) << name
                      << std::right << std::setw(10) << openclTime
                      << std::setw(10) << (serialTime / openclTime) << "x\n";
        }
        std::cout << "\n";
    }
    
    // Cleanup
    for (auto& prog : programs) clReleaseProgram(prog);
    for (auto& ctx : contexts) clReleaseContext(ctx);
    
    return 0;
}