#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_SOURCE_SIZE (0x100000)
#define ARRAY_SIZE (1024 * 1024 * 16)  // 16M elements

#define CHECK_ERROR(err, msg) \
    if (err != CL_SUCCESS) { \
        fprintf(stderr, "Error %d at %s\n", err, msg); \
        exit(1); \
    }

char* read_kernel_source(const char* filename, size_t* size) {
    FILE* fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Failed to load kernel file: %s\n", filename);
        exit(1);
    }
    
    char* source_str = (char*)malloc(MAX_SOURCE_SIZE);
    *size = fread(source_str, 1, MAX_SOURCE_SIZE, fp);
    fclose(fp);
    return source_str;
}

void get_device_name(cl_device_id device, char* name, size_t size) {
    clGetDeviceInfo(device, CL_DEVICE_NAME, size, name, NULL);
}

cl_ulong get_event_time(cl_event event, cl_profiling_info param) {
    cl_ulong time;
    clGetEventProfilingInfo(event, param, sizeof(time), &time, NULL);
    return time;
}

typedef struct {
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel kernel;
    cl_mem a_buf, b_buf, c_buf;
    cl_event kernel_event;
    char name[128];
} DeviceContext;

int main(void) {
    cl_int err;
    
    // Get all platforms
    cl_uint num_platforms;
    err = clGetPlatformIDs(0, NULL, &num_platforms);
    CHECK_ERROR(err, "clGetPlatformIDs count");
    
    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * num_platforms);
    err = clGetPlatformIDs(num_platforms, platforms, NULL);
    CHECK_ERROR(err, "clGetPlatformIDs");
    
    printf("Found %d OpenCL platform(s)\n", num_platforms);
    
    // Collect all devices from all platforms
    cl_device_id* devices = NULL;
    cl_uint total_devices = 0;
    
    for (cl_uint p = 0; p < num_platforms; p++) {
        cl_uint num_platform_devices;
        err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, 0, NULL, &num_platform_devices);
        if (err == CL_SUCCESS && num_platform_devices > 0) {
            devices = (cl_device_id*)realloc(devices, sizeof(cl_device_id) * (total_devices + num_platform_devices));
            err = clGetDeviceIDs(platforms[p], CL_DEVICE_TYPE_ALL, num_platform_devices, 
                                &devices[total_devices], NULL);
            CHECK_ERROR(err, "clGetDeviceIDs");
            total_devices += num_platform_devices;
        }
    }
    
    cl_uint num_devices = total_devices;
    printf("Found %d OpenCL device(s) total\n\n", num_devices);
    
    // Load kernel source once
    size_t source_size;
    char* source_str = read_kernel_source("vector_add.cl", &source_size);
    
    // Prepare data
    int* A = (int*)malloc(sizeof(int) * ARRAY_SIZE);
    int* B = (int*)malloc(sizeof(int) * ARRAY_SIZE);
    int* C = (int*)malloc(sizeof(int) * ARRAY_SIZE);
    
    for (int i = 0; i < ARRAY_SIZE; i++) {
        A[i] = i;
        B[i] = ARRAY_SIZE - i;
    }
    
    // Calculate work distribution
    int chunk_size = ARRAY_SIZE / num_devices;
    
    printf("=== Asynchronous Multi-Device Execution ===\n");
    printf("Total array size: %d elements\n", ARRAY_SIZE);
    printf("Chunk size per device: %d elements\n\n", chunk_size);
    
    // Create separate context for each device
    DeviceContext* dev_contexts = (DeviceContext*)malloc(sizeof(DeviceContext) * num_devices);
    
    for (cl_uint i = 0; i < num_devices; i++) {
        dev_contexts[i].device = devices[i];
        get_device_name(devices[i], dev_contexts[i].name, sizeof(dev_contexts[i].name));
        
        printf("Setting up device %d: %s\n", i, dev_contexts[i].name);
        
        // Create individual context for this device
        dev_contexts[i].context = clCreateContext(NULL, 1, &devices[i], NULL, NULL, &err);
        CHECK_ERROR(err, "clCreateContext");
        
        // Create command queue with profiling
        cl_queue_properties props[] = {CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0};
        dev_contexts[i].queue = clCreateCommandQueueWithProperties(dev_contexts[i].context, devices[i], 
                                    props, &err);
        CHECK_ERROR(err, "clCreateCommandQueue");
        
        // Create and build program for this context
        dev_contexts[i].program = clCreateProgramWithSource(dev_contexts[i].context, 1, 
                                        (const char**)&source_str, &source_size, &err);
        CHECK_ERROR(err, "clCreateProgramWithSource");
        
        err = clBuildProgram(dev_contexts[i].program, 1, &devices[i], NULL, NULL, NULL);
        if (err != CL_SUCCESS) {
            char build_log[4096];
            clGetProgramBuildInfo(dev_contexts[i].program, devices[i], CL_PROGRAM_BUILD_LOG, 
                                 sizeof(build_log), build_log, NULL);
            fprintf(stderr, "Build error for device %d:\n%s\n", i, build_log);
            exit(1);
        }
        
        // Create kernel
        dev_contexts[i].kernel = clCreateKernel(dev_contexts[i].program, "vector_add", &err);
        CHECK_ERROR(err, "clCreateKernel");
        
        // Calculate this device's chunk
        int offset = i * chunk_size;
        int size = (i == num_devices - 1) ? (ARRAY_SIZE - offset) : chunk_size;
        
        // Create buffers
        dev_contexts[i].a_buf = clCreateBuffer(dev_contexts[i].context, CL_MEM_READ_ONLY, 
                                               sizeof(int) * size, NULL, &err);
        CHECK_ERROR(err, "clCreateBuffer A");
        
        dev_contexts[i].b_buf = clCreateBuffer(dev_contexts[i].context, CL_MEM_READ_ONLY,
                                               sizeof(int) * size, NULL, &err);
        CHECK_ERROR(err, "clCreateBuffer B");
        
        dev_contexts[i].c_buf = clCreateBuffer(dev_contexts[i].context, CL_MEM_WRITE_ONLY,
                                               sizeof(int) * size, NULL, &err);
        CHECK_ERROR(err, "clCreateBuffer C");
        
        // Write input data
        err = clEnqueueWriteBuffer(dev_contexts[i].queue, dev_contexts[i].a_buf, CL_FALSE, 0,
                                   sizeof(int) * size, &A[offset], 0, NULL, NULL);
        CHECK_ERROR(err, "clEnqueueWriteBuffer A");
        
        err = clEnqueueWriteBuffer(dev_contexts[i].queue, dev_contexts[i].b_buf, CL_FALSE, 0,
                                   sizeof(int) * size, &B[offset], 0, NULL, NULL);
        CHECK_ERROR(err, "clEnqueueWriteBuffer B");
        
        // Set kernel arguments
        err = clSetKernelArg(dev_contexts[i].kernel, 0, sizeof(cl_mem), &dev_contexts[i].a_buf);
        err |= clSetKernelArg(dev_contexts[i].kernel, 1, sizeof(cl_mem), &dev_contexts[i].b_buf);
        err |= clSetKernelArg(dev_contexts[i].kernel, 2, sizeof(cl_mem), &dev_contexts[i].c_buf);
        err |= clSetKernelArg(dev_contexts[i].kernel, 3, sizeof(int), &offset);
        CHECK_ERROR(err, "clSetKernelArg");
    }
    
    printf("\nLaunching kernels on all devices simultaneously...\n");
    
    // Launch all kernels
    for (cl_uint i = 0; i < num_devices; i++) {
        int offset = i * chunk_size;
        int size = (i == num_devices - 1) ? (ARRAY_SIZE - offset) : chunk_size;
        size_t global_size = size;
        
        err = clEnqueueNDRangeKernel(dev_contexts[i].queue, dev_contexts[i].kernel, 1, NULL, 
                                     &global_size, NULL, 0, NULL, &dev_contexts[i].kernel_event);
        CHECK_ERROR(err, "clEnqueueNDRangeKernel");
    }
    
    // Wait for all to complete
    for (cl_uint i = 0; i < num_devices; i++) {
        clWaitForEvents(1, &dev_contexts[i].kernel_event);
    }
    
    printf("\n=== Execution Timeline ===\n");
    
    cl_ulong earliest_start = ULLONG_MAX;
    cl_ulong latest_end = 0;
    
    for (cl_uint i = 0; i < num_devices; i++) {
        cl_ulong start = get_event_time(dev_contexts[i].kernel_event, CL_PROFILING_COMMAND_START);
        cl_ulong end = get_event_time(dev_contexts[i].kernel_event, CL_PROFILING_COMMAND_END);
        
        printf("Device %d (%s):\n", i, dev_contexts[i].name);
        printf("  Start: %llu ns\n", start);
        printf("  End:   %llu ns\n", end);
        printf("  Duration: %.3f ms\n\n", (end - start) / 1000000.0);
        
        if (start < earliest_start) earliest_start = start;
        if (end > latest_end) latest_end = end;
    }
    
    double total_time = (latest_end - earliest_start) / 1000000.0;
    printf("Total wall-clock time: %.3f ms\n", total_time);
    
    // Analyze concurrency
    printf("\n=== Concurrency Analysis ===\n");
    for (cl_uint i = 0; i < num_devices; i++) {
        for (cl_uint j = i + 1; j < num_devices; j++) {
            cl_ulong start_i = get_event_time(dev_contexts[i].kernel_event, CL_PROFILING_COMMAND_START);
            cl_ulong end_i = get_event_time(dev_contexts[i].kernel_event, CL_PROFILING_COMMAND_END);
            cl_ulong start_j = get_event_time(dev_contexts[j].kernel_event, CL_PROFILING_COMMAND_START);
            cl_ulong end_j = get_event_time(dev_contexts[j].kernel_event, CL_PROFILING_COMMAND_END);
            
            bool overlap = !(end_i < start_j || end_j < start_i);
            printf("Device %d and %d: %s\n", i, j, 
                   overlap ? "CONCURRENT EXECUTION" : "Sequential");
        }
    }
    
    // Read results back
    for (cl_uint i = 0; i < num_devices; i++) {
        int offset = i * chunk_size;
        int size = (i == num_devices - 1) ? (ARRAY_SIZE - offset) : chunk_size;
        
        err = clEnqueueReadBuffer(dev_contexts[i].queue, dev_contexts[i].c_buf, CL_TRUE, 0,
                                 sizeof(int) * size, &C[offset], 0, NULL, NULL);
        CHECK_ERROR(err, "clEnqueueReadBuffer");
    }
    
    // Verify
    printf("\n=== Verification (first 10 elements) ===\n");
    bool correct = true;
    for (int i = 0; i < 10; i++) {
        int expected = A[i] + B[i];
        if (C[i] != expected) {
            printf("Error at %d: expected %d, got %d\n", i, expected, C[i]);
            correct = false;
        } else {
            printf("%d + %d = %d\n", A[i], B[i], C[i]);
        }
    }
    printf(correct ? "\nVerification PASSED\n" : "\nVerification FAILED\n");
    
    // Cleanup
    for (cl_uint i = 0; i < num_devices; i++) {
        clReleaseMemObject(dev_contexts[i].a_buf);
        clReleaseMemObject(dev_contexts[i].b_buf);
        clReleaseMemObject(dev_contexts[i].c_buf);
        clReleaseKernel(dev_contexts[i].kernel);
        clReleaseProgram(dev_contexts[i].program);
        clReleaseCommandQueue(dev_contexts[i].queue);
        clReleaseContext(dev_contexts[i].context);
        clReleaseEvent(dev_contexts[i].kernel_event);
    }
    
    free(dev_contexts);
    free(platforms);
    free(devices);
    free(A);
    free(B);
    free(C);
    free(source_str);
    
    printf("\nPress any key to exit...\n");
    getchar();
    
    return 0;
}