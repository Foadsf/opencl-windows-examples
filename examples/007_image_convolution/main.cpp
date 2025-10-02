#define CL_TARGET_OPENCL_VERSION 300
#define _USE_MATH_DEFINES
#include <CL/opencl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <execution>
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

// Generate Gaussian kernel
std::vector<float> createGaussianKernel(int size, float sigma) {
    std::vector<float> kernel(size * size);
    int half = size / 2;
    float sum = 0.0f;
    
    for (int y = -half; y <= half; y++) {
        for (int x = -half; x <= half; x++) {
            float value = std::exp(-(x*x + y*y) / (2.0f * sigma * sigma));
            kernel[(y + half) * size + (x + half)] = value;
            sum += value;
        }
    }
    
    // Normalize
    for (auto& v : kernel) v /= sum;
    return kernel;
}

// Generate 1D Gaussian kernel (for separable convolution)
std::vector<float> createGaussianKernel1D(int size, float sigma) {
    std::vector<float> kernel(size);
    int half = size / 2;
    float sum = 0.0f;
    
    for (int i = -half; i <= half; i++) {
        float value = std::exp(-(i*i) / (2.0f * sigma * sigma));
        kernel[i + half] = value;
        sum += value;
    }
    
    for (auto& v : kernel) v /= sum;
    return kernel;
}

// 1. Serial implementation
double convolveSerial(const std::vector<float>& input,
                      std::vector<float>& output,
                      const std::vector<float>& kernel,
                      int width, int height, int ksize) {
    auto start = std::chrono::high_resolution_clock::now();
    
    int khalf = ksize / 2;
    
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            
            for (int ky = -khalf; ky <= khalf; ky++) {
                for (int kx = -khalf; kx <= khalf; kx++) {
                    int ix = std::max(0, std::min(x + kx, width - 1));
                    int iy = std::max(0, std::min(y + ky, height - 1));
                    
                    int kidx = (ky + khalf) * ksize + (kx + khalf);
                    sum += input[iy * width + ix] * kernel[kidx];
                }
            }
            
            output[y * width + x] = sum;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// 2. OpenMP implementation
double convolveOpenMP(const std::vector<float>& input,
                      std::vector<float>& output,
                      const std::vector<float>& kernel,
                      int width, int height, int ksize) {
    auto start = std::chrono::high_resolution_clock::now();
    
    int khalf = ksize / 2;
    
    #pragma omp parallel for
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            float sum = 0.0f;
            
            for (int ky = -khalf; ky <= khalf; ky++) {
                for (int kx = -khalf; kx <= khalf; kx++) {
                    int ix = std::max(0, std::min(x + kx, width - 1));
                    int iy = std::max(0, std::min(y + ky, height - 1));
                    
                    int kidx = (ky + khalf) * ksize + (kx + khalf);
                    sum += input[iy * width + ix] * kernel[kidx];
                }
            }
            
            output[y * width + x] = sum;
        }
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// 3. OpenCL implementation
double convolveOpenCL(const std::vector<float>& input,
                      std::vector<float>& output,
                      const std::vector<float>& kernel,
                      int width, int height, int ksize,
                      cl_device_id device,
                      cl_context context,
                      cl_program program,
                      const char* kernelName,
                      bool useLocal = false) {
    cl_int err;
    
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    checkError(err, "clCreateCommandQueue");
    
    size_t imageSize = width * height * sizeof(float);
    size_t kernelSize = ksize * ksize * sizeof(float);
    
    cl_mem bufInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      imageSize, (void*)input.data(), &err);
    checkError(err, "clCreateBuffer input");
    
    cl_mem bufOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imageSize, nullptr, &err);
    checkError(err, "clCreateBuffer output");
    
    cl_mem bufKernel = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       kernelSize, (void*)kernel.data(), &err);
    checkError(err, "clCreateBuffer kernel");
    
    cl_kernel clKernel = clCreateKernel(program, kernelName, &err);
    checkError(err, "clCreateKernel");
    
    clSetKernelArg(clKernel, 0, sizeof(cl_mem), &bufInput);
    clSetKernelArg(clKernel, 1, sizeof(cl_mem), &bufOutput);
    clSetKernelArg(clKernel, 2, sizeof(cl_mem), &bufKernel);
    clSetKernelArg(clKernel, 3, sizeof(int), &width);
    clSetKernelArg(clKernel, 4, sizeof(int), &height);
    clSetKernelArg(clKernel, 5, sizeof(int), &ksize);
    
    if (useLocal) {
        const int LOCAL_SIZE = 16;
        int khalf = ksize / 2;
        int tileSize = (LOCAL_SIZE + 2 * khalf) * (LOCAL_SIZE + 2 * khalf);
        clSetKernelArg(clKernel, 6, tileSize * sizeof(float), nullptr);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (useLocal) {
        const int LOCAL_SIZE = 16;
        size_t globalSize[2] = {(size_t)((width + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE,
                                (size_t)((height + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE};
        size_t localSize[2] = {LOCAL_SIZE, LOCAL_SIZE};
        err = clEnqueueNDRangeKernel(queue, clKernel, 2, nullptr, globalSize, localSize, 0, nullptr, nullptr);
    } else {
        size_t globalSize[2] = {(size_t)width, (size_t)height};
        err = clEnqueueNDRangeKernel(queue, clKernel, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    }
    checkError(err, "clEnqueueNDRangeKernel");
    
    clFinish(queue);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    clEnqueueReadBuffer(queue, bufOutput, CL_TRUE, 0, imageSize, output.data(), 0, nullptr, nullptr);
    
    clReleaseMemObject(bufInput);
    clReleaseMemObject(bufOutput);
    clReleaseMemObject(bufKernel);
    clReleaseKernel(clKernel);
    clReleaseCommandQueue(queue);
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// Separable convolution (OpenCL)
double convolveSeparable(const std::vector<float>& input,
                         std::vector<float>& output,
                         const std::vector<float>& kernel1d,
                         int width, int height, int ksize,
                         cl_device_id device,
                         cl_context context,
                         cl_program program) {
    cl_int err;
    
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    
    size_t imageSize = width * height * sizeof(float);
    size_t kernelSize = ksize * sizeof(float);
    
    cl_mem bufInput = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                      imageSize, (void*)input.data(), &err);
    cl_mem bufTemp = clCreateBuffer(context, CL_MEM_READ_WRITE, imageSize, nullptr, &err);
    cl_mem bufOutput = clCreateBuffer(context, CL_MEM_WRITE_ONLY, imageSize, nullptr, &err);
    cl_mem bufKernel = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                       kernelSize, (void*)kernel1d.data(), &err);
    
    cl_kernel kernelH = clCreateKernel(program, "convolve_h", &err);
    cl_kernel kernelV = clCreateKernel(program, "convolve_v", &err);
    
    // Set args for horizontal pass
    clSetKernelArg(kernelH, 0, sizeof(cl_mem), &bufInput);
    clSetKernelArg(kernelH, 1, sizeof(cl_mem), &bufTemp);
    clSetKernelArg(kernelH, 2, sizeof(cl_mem), &bufKernel);
    clSetKernelArg(kernelH, 3, sizeof(int), &width);
    clSetKernelArg(kernelH, 4, sizeof(int), &height);
    clSetKernelArg(kernelH, 5, sizeof(int), &ksize);
    
    // Set args for vertical pass
    clSetKernelArg(kernelV, 0, sizeof(cl_mem), &bufTemp);
    clSetKernelArg(kernelV, 1, sizeof(cl_mem), &bufOutput);
    clSetKernelArg(kernelV, 2, sizeof(cl_mem), &bufKernel);
    clSetKernelArg(kernelV, 3, sizeof(int), &width);
    clSetKernelArg(kernelV, 4, sizeof(int), &height);
    clSetKernelArg(kernelV, 5, sizeof(int), &ksize);
    
    size_t globalSize[2] = {(size_t)width, (size_t)height};
    
    auto start = std::chrono::high_resolution_clock::now();
    
    clEnqueueNDRangeKernel(queue, kernelH, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    clEnqueueNDRangeKernel(queue, kernelV, 2, nullptr, globalSize, nullptr, 0, nullptr, nullptr);
    
    clFinish(queue);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    clEnqueueReadBuffer(queue, bufOutput, CL_TRUE, 0, imageSize, output.data(), 0, nullptr, nullptr);
    
    clReleaseMemObject(bufInput);
    clReleaseMemObject(bufTemp);
    clReleaseMemObject(bufOutput);
    clReleaseMemObject(bufKernel);
    clReleaseKernel(kernelH);
    clReleaseKernel(kernelV);
    clReleaseCommandQueue(queue);
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "=== Image Convolution Performance Comparison ===\n\n";
    
    // Test configurations
    std::vector<int> imageSizes = {512, 1024, 2048, 4096};
    std::vector<int> kernelSizes = {3, 5, 7, 11, 15};
    
    // Get OpenCL devices
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    
    std::vector<cl_device_id> devices;
    std::vector<std::string> deviceNames;
    std::vector<cl_context> contexts;
    std::vector<cl_program> programs;
    
    std::string kernelSource = loadKernelSource("convolution.cl");
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
                
                // CHECK BUILD STATUS
                if (err != CL_SUCCESS) {
                    size_t logSize;
                    clGetProgramBuildInfo(program, platformDevices[d], CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
                    std::vector<char> log(logSize);
                    clGetProgramBuildInfo(program, platformDevices[d], CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
                    std::cerr << "Build error for " << name << ":\n" << log.data() << "\n";
                    std::cerr << "Skipping this device.\n\n";
                    clReleaseContext(context);
                    contexts.pop_back();
                    devices.pop_back();
                    deviceNames.pop_back();
                    continue;
                }
                
                programs.push_back(program);
            }
        }
    }
    
    std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n";
    std::cout << "OpenCL devices: " << devices.size() << "\n\n";
    
    // Test specific configuration
    for (int imgSize : imageSizes) {
        for (int ksize : kernelSizes) {
            int width = imgSize;
            int height = imgSize;
            
            std::cout << "========================================\n";
            std::cout << "Image: " << width << "x" << height << ", Kernel: " << ksize << "x" << ksize << "\n";
            std::cout << "Operations per pixel: " << (ksize * ksize) << "\n";
            std::cout << "Total operations: " << (width * height * ksize * ksize / 1e6) << " million\n";
            std::cout << "========================================\n";
            
            // Create synthetic image
            std::vector<float> input(width * height);
            for (size_t i = 0; i < input.size(); i++) {
                input[i] = static_cast<float>(i % 256) / 255.0f;
            }
            
            std::vector<float> output(width * height);
            std::vector<float> kernel2d = createGaussianKernel(ksize, ksize / 6.0f);
            std::vector<float> kernel1d = createGaussianKernel1D(ksize, ksize / 6.0f);
            
            // Serial
            double serialTime = convolveSerial(input, output, kernel2d, width, height, ksize);
            std::vector<float> expectedResult = output;
            
            // OpenMP
            std::fill(output.begin(), output.end(), 0.0f);
            double openmpTime = convolveOpenMP(input, output, kernel2d, width, height, ksize);
            
            std::cout << "\n" << std::fixed << std::setprecision(2);
            std::cout << std::left << std::setw(40) << "Implementation"
                      << std::right << std::setw(12) << "Time (ms)"
                      << std::setw(12) << "Speedup\n";
            std::cout << std::string(64, '-') << "\n";
            
            std::cout << std::left << std::setw(40) << "Serial C++"
                      << std::right << std::setw(12) << serialTime
                      << std::setw(12) << "1.00x\n";
            
            std::cout << std::left << std::setw(40) << "OpenMP"
                      << std::right << std::setw(12) << openmpTime
                      << std::setw(12) << (serialTime / openmpTime) << "x\n";
            
            // OpenCL devices
            for (size_t i = 0; i < devices.size(); i++) {
                // Simple version
                std::fill(output.begin(), output.end(), 0.0f);
                double simpleTime = convolveOpenCL(input, output, kernel2d, width, height, ksize,
                                                    devices[i], contexts[i], programs[i],
                                                    "convolve_2d", false);
                
                std::string name = "OpenCL: " + deviceNames[i].substr(0, 22);
                std::cout << std::left << std::setw(40) << name
                          << std::right << std::setw(12) << simpleTime
                          << std::setw(12) << (serialTime / simpleTime) << "x\n";
                
                // Local memory version
                std::fill(output.begin(), output.end(), 0.0f);
                double localTime = convolveOpenCL(input, output, kernel2d, width, height, ksize,
                                                   devices[i], contexts[i], programs[i],
                                                   "convolve_2d_local", true);
                
                std::string localName = "OpenCL: " + deviceNames[i].substr(0, 18) + " (local)";
                std::cout << std::left << std::setw(40) << localName
                          << std::right << std::setw(12) << localTime
                          << std::setw(12) << (serialTime / localTime) << "x\n";
                
                // Separable version
                std::fill(output.begin(), output.end(), 0.0f);
                double sepTime = convolveSeparable(input, output, kernel1d, width, height, ksize,
                                                    devices[i], contexts[i], programs[i]);
                
                std::string sepName = "OpenCL: " + deviceNames[i].substr(0, 16) + " (separable)";
                std::cout << std::left << std::setw(40) << sepName
                          << std::right << std::setw(12) << sepTime
                          << std::setw(12) << (serialTime / sepTime) << "x\n";
            }
            
            std::cout << "\n";
        }
    }
    
    // Cleanup
    for (auto& prog : programs) clReleaseProgram(prog);
    for (auto& ctx : contexts) clReleaseContext(ctx);
    
    return 0;
}