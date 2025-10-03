#define CL_TARGET_OPENCL_VERSION 300
#define _USE_MATH_DEFINES
#include <CL/opencl.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cmath>
#include <omp.h>

struct Body {
    float x, y, z, w;  // position (w unused, for alignment)
    float vx, vy, vz, vw;  // velocity
    float mass;
};

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

// Initialize random particle system
void initializeBodies(std::vector<Body>& bodies, int n) {
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> pos_dist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> vel_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> mass_dist(1.0f, 10.0f);
    
    for (int i = 0; i < n; i++) {
        bodies[i].x = pos_dist(gen);
        bodies[i].y = pos_dist(gen);
        bodies[i].z = pos_dist(gen);
        bodies[i].w = 0.0f;
        
        bodies[i].vx = vel_dist(gen);
        bodies[i].vy = vel_dist(gen);
        bodies[i].vz = vel_dist(gen);
        bodies[i].vw = 0.0f;
        
        bodies[i].mass = mass_dist(gen);
    }
}

// Serial N-body force calculation
double computeForcesSerial(const std::vector<Body>& bodies,
                           std::vector<float>& acc_x,
                           std::vector<float>& acc_y,
                           std::vector<float>& acc_z,
                           float softening) {
    int n = bodies.size();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int i = 0; i < n; i++) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            
            float dx = bodies[j].x - bodies[i].x;
            float dy = bodies[j].y - bodies[i].y;
            float dz = bodies[j].z - bodies[i].z;
            
            float dist_sq = dx*dx + dy*dy + dz*dz + softening*softening;
            float dist = std::sqrt(dist_sq);
            float dist_cubed = dist_sq * dist;
            float force = bodies[j].mass / dist_cubed;
            
            ax += dx * force;
            ay += dy * force;
            az += dz * force;
        }
        
        acc_x[i] = ax;
        acc_y[i] = ay;
        acc_z[i] = az;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// OpenMP N-body
double computeForcesOpenMP(const std::vector<Body>& bodies,
                           std::vector<float>& acc_x,
                           std::vector<float>& acc_y,
                           std::vector<float>& acc_z,
                           float softening) {
    int n = bodies.size();
    
    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f;
        
        for (int j = 0; j < n; j++) {
            if (i == j) continue;
            
            float dx = bodies[j].x - bodies[i].x;
            float dy = bodies[j].y - bodies[i].y;
            float dz = bodies[j].z - bodies[i].z;
            
            float dist_sq = dx*dx + dy*dy + dz*dz + softening*softening;
            float dist = std::sqrt(dist_sq);
            float dist_cubed = dist_sq * dist;
            float force = bodies[j].mass / dist_cubed;
            
            ax += dx * force;
            ay += dy * force;
            az += dz * force;
        }
        
        acc_x[i] = ax;
        acc_y[i] = ay;
        acc_z[i] = az;
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

// OpenCL N-body
double computeForcesOpenCL(const std::vector<Body>& bodies,
                           std::vector<float>& acc_x,
                           std::vector<float>& acc_y,
                           std::vector<float>& acc_z,
                           float softening,
                           cl_device_id device,
                           cl_context context,
                           cl_program program,
                           bool useTiled) {
    cl_int err;
    int n = bodies.size();
    
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, nullptr, &err);
    checkError(err, "clCreateCommandQueue");
    
    // Prepare data
    std::vector<float> positions(n * 4);
    std::vector<float> masses(n);
    std::vector<float> accelerations(n * 4);
    
    for (int i = 0; i < n; i++) {
        positions[i*4 + 0] = bodies[i].x;
        positions[i*4 + 1] = bodies[i].y;
        positions[i*4 + 2] = bodies[i].z;
        positions[i*4 + 3] = 0.0f;
        masses[i] = bodies[i].mass;
    }
    
    cl_mem bufPos = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                    n * 4 * sizeof(float), positions.data(), &err);
    checkError(err, "clCreateBuffer positions");
    
    cl_mem bufMass = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                                     n * sizeof(float), masses.data(), &err);
    checkError(err, "clCreateBuffer masses");
    
    cl_mem bufAcc = clCreateBuffer(context, CL_MEM_WRITE_ONLY, n * 4 * sizeof(float), nullptr, &err);
    checkError(err, "clCreateBuffer accelerations");
    
    const char* kernelName = useTiled ? "compute_forces_tiled" : "compute_forces";
    cl_kernel kernel = clCreateKernel(program, kernelName, &err);
    checkError(err, "clCreateKernel");
    
    clSetKernelArg(kernel, 0, sizeof(cl_mem), &bufPos);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), &bufMass);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), &bufAcc);
    clSetKernelArg(kernel, 3, sizeof(int), &n);
    clSetKernelArg(kernel, 4, sizeof(float), &softening);
    
    if (useTiled) {
        const int LOCAL_SIZE = 256;
        clSetKernelArg(kernel, 5, LOCAL_SIZE * 4 * sizeof(float), nullptr);
        clSetKernelArg(kernel, 6, LOCAL_SIZE * sizeof(float), nullptr);
    }
    
    auto start = std::chrono::high_resolution_clock::now();
    
    if (useTiled) {
        const int LOCAL_SIZE = 256;
        size_t globalSize = ((n + LOCAL_SIZE - 1) / LOCAL_SIZE) * LOCAL_SIZE;
        size_t localSize = LOCAL_SIZE;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, nullptr);
    } else {
        size_t globalSize = n;
        err = clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, nullptr, 0, nullptr, nullptr);
    }
    checkError(err, "clEnqueueNDRangeKernel");
    
    clFinish(queue);
    
    auto end = std::chrono::high_resolution_clock::now();
    
    clEnqueueReadBuffer(queue, bufAcc, CL_TRUE, 0, n * 4 * sizeof(float),
                        accelerations.data(), 0, nullptr, nullptr);
    
    for (int i = 0; i < n; i++) {
        acc_x[i] = accelerations[i*4 + 0];
        acc_y[i] = accelerations[i*4 + 1];
        acc_z[i] = accelerations[i*4 + 2];
    }
    
    clReleaseMemObject(bufPos);
    clReleaseMemObject(bufMass);
    clReleaseMemObject(bufAcc);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main() {
    std::cout << "=== N-Body Simulation Performance Comparison ===\n\n";
    
    std::vector<int> bodyCounts = {128, 256, 512, 1024, 2048, 4096};
    const float softening = 0.1f;
    
    // Get OpenCL devices
    cl_uint numPlatforms;
    clGetPlatformIDs(0, nullptr, &numPlatforms);
    std::vector<cl_platform_id> platforms(numPlatforms);
    clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    
    std::vector<cl_device_id> devices;
    std::vector<std::string> deviceNames;
    std::vector<cl_context> contexts;
    std::vector<cl_program> programs;
    
    std::string kernelSource = loadKernelSource("nbody.cl");
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
    
    for (int n : bodyCounts) {
        std::cout << "========================================\n";
        std::cout << "N-Body with " << n << " particles\n";
        std::cout << "Force calculations: " << (n * (n-1)) << " (O(n²))\n";
        std::cout << "========================================\n";
        
        std::vector<Body> bodies(n);
        initializeBodies(bodies, n);
        
        std::vector<float> acc_x(n), acc_y(n), acc_z(n);
        
        // Serial
        double serialTime = computeForcesSerial(bodies, acc_x, acc_y, acc_z, softening);
        
        // OpenMP
        double openmpTime = computeForcesOpenMP(bodies, acc_x, acc_y, acc_z, softening);
        
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
            double simpleTime = computeForcesOpenCL(bodies, acc_x, acc_y, acc_z, softening,
                                                     devices[i], contexts[i], programs[i], false);
            
            std::string name = "OpenCL: " + deviceNames[i].substr(0, 22);
            std::cout << std::left << std::setw(40) << name
                      << std::right << std::setw(12) << simpleTime
                      << std::setw(12) << (serialTime / simpleTime) << "x\n";
            
            double tiledTime = computeForcesOpenCL(bodies, acc_x, acc_y, acc_z, softening,
                                                    devices[i], contexts[i], programs[i], true);
            
            std::string tiledName = "OpenCL: " + deviceNames[i].substr(0, 18) + " (tiled)";
            std::cout << std::left << std::setw(40) << tiledName
                      << std::right << std::setw(12) << tiledTime
                      << std::setw(12) << (serialTime / tiledTime) << "x\n";
        }
        
        std::cout << "\n";
    }
    
    // Cleanup
    for (auto& prog : programs) clReleaseProgram(prog);
    for (auto& ctx : contexts) clReleaseContext(ctx);
    
    return 0;
}