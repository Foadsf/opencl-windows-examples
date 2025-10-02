#define CL_TARGET_OPENCL_VERSION 300
#include <CL/opencl.h>
#include <iostream>
#include <vector>
#include <cstring>

int main() {
    std::cout << "=== OpenCL Device Enumeration ===\n\n";

    cl_uint numPlatforms;
    cl_int err = clGetPlatformIDs(0, nullptr, &numPlatforms);
    if (err != CL_SUCCESS) {
        std::cerr << "Error getting platform count: " << err << "\n";
        return 1;
    }
    
    if (numPlatforms == 0) {
        std::cerr << "No OpenCL platforms found!\n";
        return 1;
    }

    std::cout << "Found " << numPlatforms << " OpenCL platform(s)\n\n";

    std::vector<cl_platform_id> platforms(numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms.data(), nullptr);
    if (err != CL_SUCCESS) {
        std::cerr << "Error retrieving platforms: " << err << "\n";
        return 1;
    }

    for (cl_uint i = 0; i < numPlatforms; i++) {
        std::cout << "Platform " << i << ":\n";
        std::cout.flush();
        
        char buffer[1024];
        std::memset(buffer, 0, sizeof(buffer));
        
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, sizeof(buffer), buffer, nullptr);
        std::cout << "  Name: " << (err == CL_SUCCESS ? buffer : "<error>") << "\n";
        std::cout.flush();
        
        std::memset(buffer, 0, sizeof(buffer));
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_VENDOR, sizeof(buffer), buffer, nullptr);
        std::cout << "  Vendor: " << (err == CL_SUCCESS ? buffer : "<error>") << "\n";
        std::cout.flush();
        
        std::memset(buffer, 0, sizeof(buffer));
        err = clGetPlatformInfo(platforms[i], CL_PLATFORM_VERSION, sizeof(buffer), buffer, nullptr);
        std::cout << "  Version: " << (err == CL_SUCCESS ? buffer : "<error>") << "\n";
        std::cout.flush();

        std::cout << "  Attempting device enumeration...\n";
        std::cout.flush();

        cl_uint numDevices = 0;
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &numDevices);
        if (err != CL_SUCCESS) {
            std::cout << "  Devices: <error " << err << ", skipping>\n\n";
            std::cout.flush();
            continue;
        }
        std::cout << "  Devices: " << numDevices << "\n";
        std::cout.flush();

        if (numDevices == 0) {
            std::cout << "\n";
            continue;
        }

        std::vector<cl_device_id> devices(numDevices);
        err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, numDevices, devices.data(), nullptr);
        if (err != CL_SUCCESS) {
            std::cout << "  <cannot retrieve device list>\n\n";
            std::cout.flush();
            continue;
        }

        for (cl_uint j = 0; j < numDevices; j++) {
            std::cout << "    Device " << j << ":\n";
            std::cout.flush();
            
            std::memset(buffer, 0, sizeof(buffer));
            err = clGetDeviceInfo(devices[j], CL_DEVICE_NAME, sizeof(buffer), buffer, nullptr);
            std::cout << "      Name: " << (err == CL_SUCCESS ? buffer : "<error>") << "\n";
            std::cout.flush();
            
            cl_device_type type;
            err = clGetDeviceInfo(devices[j], CL_DEVICE_TYPE, sizeof(type), &type, nullptr);
            if (err == CL_SUCCESS) {
                std::cout << "      Type: ";
                if (type & CL_DEVICE_TYPE_CPU) std::cout << "CPU ";
                if (type & CL_DEVICE_TYPE_GPU) std::cout << "GPU ";
                if (type & CL_DEVICE_TYPE_ACCELERATOR) std::cout << "ACCELERATOR ";
                std::cout << "\n";
                std::cout.flush();
            }

            cl_ulong globalMem;
            err = clGetDeviceInfo(devices[j], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(globalMem), &globalMem, nullptr);
            if (err == CL_SUCCESS) {
                std::cout << "      Global Memory: " << (globalMem / (1024 * 1024)) << " MB\n";
                std::cout.flush();
            }

            cl_uint computeUnits;
            err = clGetDeviceInfo(devices[j], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(computeUnits), &computeUnits, nullptr);
            if (err == CL_SUCCESS) {
                std::cout << "      Compute Units: " << computeUnits << "\n";
                std::cout.flush();
            }
        }
        std::cout << "\n";
        std::cout.flush();
    }

    std::cout << "Enumeration complete!\n";
    return 0;
}