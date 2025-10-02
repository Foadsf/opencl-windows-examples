// Clamp function for boundary handling
inline int clamp_int(int val, int min_val, int max_val) {
    if (val < min_val) return min_val;
    if (val > max_val) return max_val;
    return val;
}

// Simple 2D convolution kernel
__kernel void convolve_2d(__global const float* input,
                          __global float* output,
                          __constant float* filter,
                          const int width,
                          const int height,
                          const int ksize)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int khalf = ksize / 2;
    float sum = 0.0f;
    
    for (int ky = -khalf; ky <= khalf; ky++) {
        for (int kx = -khalf; kx <= khalf; kx++) {
            int ix = x + kx;
            int iy = y + ky;
            
            // Clamp to image boundaries
            ix = clamp_int(ix, 0, width - 1);
            iy = clamp_int(iy, 0, height - 1);
            
            int kidx = (ky + khalf) * ksize + (kx + khalf);
            sum += input[iy * width + ix] * filter[kidx];
        }
    }
    
    output[y * width + x] = sum;
}

// Optimized version with local memory
__kernel void convolve_2d_local(__global const float* input,
                                __global float* output,
                                __constant float* filter,
                                const int width,
                                const int height,
                                const int ksize,
                                __local float* tile)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int lw = get_local_size(0);
    int lh = get_local_size(1);
    
    int khalf = ksize / 2;
    int tile_w = lw + 2 * khalf;
    int tile_h = lh + 2 * khalf;
    
    // Load tile with halo into local memory
    for (int ty = ly; ty < tile_h; ty += lh) {
        for (int tx = lx; tx < tile_w; tx += lw) {
            int ix = gx - khalf + tx - lx;
            int iy = gy - khalf + ty - ly;
            
            // Clamp to boundaries
            ix = clamp_int(ix, 0, width - 1);
            iy = clamp_int(iy, 0, height - 1);
            
            tile[ty * tile_w + tx] = input[iy * width + ix];
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (gx >= width || gy >= height) return;
    
    // Perform convolution using local memory
    float sum = 0.0f;
    for (int ky = 0; ky < ksize; ky++) {
        for (int kx = 0; kx < ksize; kx++) {
            int tx = lx + kx;
            int ty = ly + ky;
            sum += tile[ty * tile_w + tx] * filter[ky * ksize + kx];
        }
    }
    
    output[gy * width + gx] = sum;
}

// Separable convolution (horizontal pass)
__kernel void convolve_h(__global const float* input,
                         __global float* output,
                         __constant float* filter,
                         const int width,
                         const int height,
                         const int ksize)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int khalf = ksize / 2;
    float sum = 0.0f;
    
    for (int k = -khalf; k <= khalf; k++) {
        int ix = clamp_int(x + k, 0, width - 1);
        sum += input[y * width + ix] * filter[k + khalf];
    }
    
    output[y * width + x] = sum;
}

// Separable convolution (vertical pass)
__kernel void convolve_v(__global const float* input,
                         __global float* output,
                         __constant float* filter,
                         const int width,
                         const int height,
                         const int ksize)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    
    if (x >= width || y >= height) return;
    
    int khalf = ksize / 2;
    float sum = 0.0f;
    
    for (int k = -khalf; k <= khalf; k++) {
        int iy = clamp_int(y + k, 0, height - 1);
        sum += input[iy * width + x] * filter[k + khalf];
    }
    
    output[y * width + x] = sum;
}