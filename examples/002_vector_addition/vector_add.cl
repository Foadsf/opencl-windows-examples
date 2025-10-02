__kernel void vector_add(__global const float* a,
                         __global const float* b,
                         __global float* result,
                         const unsigned int n)
{
    int gid = get_global_id(0);
    if (gid < n) {
        result[gid] = a[gid] + b[gid];
    }
}