// Simple matrix multiplication: C = A * B
__kernel void matrix_multiply(__global const float* A,
                              __global const float* B,
                              __global float* C,
                              const int M,  // rows of A
                              const int N,  // cols of A = rows of B
                              const int K)  // cols of B
{
    int row = get_global_id(0);
    int col = get_global_id(1);
    
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[row * N + i] * B[i * K + col];
        }
        C[row * K + col] = sum;
    }
}

// Optimized version with local memory tiling
__kernel void matrix_multiply_tiled(__global const float* A,
                                    __global const float* B,
                                    __global float* C,
                                    const int M,
                                    const int N,
                                    const int K,
                                    __local float* A_tile,
                                    __local float* B_tile)
{
    const int TILE_SIZE = 16;
    
    int globalRow = get_global_id(0);
    int globalCol = get_global_id(1);
    int localRow = get_local_id(0);
    int localCol = get_local_id(1);
    
    float sum = 0.0f;
    
    int numTiles = (N + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load tiles into local memory
        int tiledRow = TILE_SIZE * t + localCol;
        int tiledCol = TILE_SIZE * t + localRow;
        
        A_tile[localRow * TILE_SIZE + localCol] = 
            (globalRow < M && tiledRow < N) ? A[globalRow * N + tiledRow] : 0.0f;
        
        B_tile[localRow * TILE_SIZE + localCol] = 
            (tiledCol < N && globalCol < K) ? B[tiledCol * K + globalCol] : 0.0f;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += A_tile[localRow * TILE_SIZE + k] * B_tile[k * TILE_SIZE + localCol];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (globalRow < M && globalCol < K) {
        C[globalRow * K + globalCol] = sum;
    }
}