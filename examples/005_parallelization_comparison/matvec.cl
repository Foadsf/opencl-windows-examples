__kernel void matvec_multiply(__global const float* matrix,
                              __global const float* vector,
                              __global float* result,
                              const int rows,
                              const int cols)
{
    int i = get_global_id(0);
    if (i < rows) {
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += matrix[i * cols + j] * vector[j];
        }
        result[i] = sum;
    }
}