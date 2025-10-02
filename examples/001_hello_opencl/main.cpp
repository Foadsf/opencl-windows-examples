#define CL_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>

std::string loadKernelSource(const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open kernel file: " << filename << "\n";
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

int main() {
    std::cout << "=== OpenCL Hello World ===\n\n";

    // Get platform (use NVIDIA for reliability)
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);

    cl_platform_id selectedPlatform = platforms[0];
    char platformName[128];
    clGetPlatformInfo(selectedPlatform, CL_PLATFORM_NAME, sizeof(platformName), platformName, nullptr);
    std::cout << "Using platform: " << platformName << "\n";

    // Get GPU device
    cl_device_id device;
    cl_int err = clGetDeviceIDs(selectedPlatform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr);
    checkError(err, "clGetDeviceIDs");

    char deviceName[128];
    clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, nullptr);
    std::cout << "Using device: " << deviceName << "\n\n";

    // Create context
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
    checkError(err, "clCreateContext");

    // Create command queue
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    checkError(err, "clCreateCommandQueueWithProperties");

    // Load and build kernel
    std::string kernelSource = loadKernelSource("hello.cl");
    if (kernelSource.empty()) {
        return 1;
    }

    const char* kernelSourcePtr = kernelSource.c_str();
    size_t kernelSourceSize = kernelSource.size();

    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourcePtr, &kernelSourceSize, &err);
    checkError(err, "clCreateProgramWithSource");

    err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
    if (err != CL_SUCCESS) {
        size_t logSize;
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
        std::vector<char> log(logSize);
        clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, log.data(), nullptr);
        std::cerr << "Build error:\n" << log.data() << "\n";
        return 1;
    }

    cl_kernel kernel = clCreateKernel(program, "hello_kernel", &err);
    checkError(err, "clCreateKernel");

    // Create buffer for message
    const size_t messageSize = 16;
    cl_mem messageBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, messageSize, nullptr, &err);
    checkError(err, "clCreateBuffer");

    // Set kernel argument
    err = clSetKernelArg(kernel, 0, sizeof(cl_mem), &messageBuffer);
    checkError(err, "clSetKernelArg");

    // Execute kernel
    size_t globalWorkSize = 1;
    std::cout << "Executing kernel...\n";
    err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalWorkSize, nullptr, 0, nullptr, nullptr);
    checkError(err, "clEnqueueNDRangeKernel");

    // Read result
    char message[messageSize];
    err = clEnqueueReadBuffer(queue, messageBuffer, CL_TRUE, 0, messageSize, message, 0, nullptr, nullptr);
    checkError(err, "clEnqueueReadBuffer");

    std::cout << "Kernel output: " << message << "\n\n";

    // Cleanup
    clReleaseMemObject(messageBuffer);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    std::cout << "Success!\n";
    return 0;
}