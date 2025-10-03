// Simple N-body kernel - each thread computes forces on one body
__kernel void compute_forces(__global const float4* positions,
                             __global const float* masses,
                             __global float4* accelerations,
                             const int n,
                             const float softening)
{
    int i = get_global_id(0);
    if (i >= n) return;
    
    float4 pos_i = positions[i];
    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    
    // Compute force from all other bodies
    for (int j = 0; j < n; j++) {
        if (i == j) continue;
        
        float4 pos_j = positions[j];
        float4 r = pos_j - pos_i;
        
        // Distance with softening
        float dist_sq = r.x * r.x + r.y * r.y + r.z * r.z + softening * softening;
        float dist = sqrt(dist_sq);
        float dist_cubed = dist_sq * dist;
        
        // Newton's law: F = G * m1 * m2 / r^2
        // a = F / m1 = G * m2 / r^2
        float force = masses[j] / dist_cubed;
        
        acc += r * force;
    }
    
    accelerations[i] = acc;
}

// Optimized version using local memory tiling
__kernel void compute_forces_tiled(__global const float4* positions,
                                   __global const float* masses,
                                   __global float4* accelerations,
                                   const int n,
                                   const float softening,
                                   __local float4* shared_pos,
                                   __local float* shared_mass)
{
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    
    float4 acc = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 pos_i = (global_id < n) ? positions[global_id] : (float4)(0.0f);
    
    // Process bodies in tiles
    int num_tiles = (n + local_size - 1) / local_size;
    
    for (int tile = 0; tile < num_tiles; tile++) {
        int j = tile * local_size + local_id;
        
        // Load tile into local memory
        if (j < n) {
            shared_pos[local_id] = positions[j];
            shared_mass[local_id] = masses[j];
        } else {
            shared_pos[local_id] = (float4)(0.0f);
            shared_mass[local_id] = 0.0f;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        // Compute forces from bodies in this tile
        if (global_id < n) {
            for (int k = 0; k < local_size; k++) {
                int j_global = tile * local_size + k;
                if (j_global >= n || j_global == global_id) continue;
                
                float4 r = shared_pos[k] - pos_i;
                float dist_sq = r.x * r.x + r.y * r.y + r.z * r.z + softening * softening;
                float dist = sqrt(dist_sq);
                float dist_cubed = dist_sq * dist;
                float force = shared_mass[k] / dist_cubed;
                
                acc += r * force;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if (global_id < n) {
        accelerations[global_id] = acc;
    }
}

// Integration kernel - update positions and velocities
__kernel void integrate(__global float4* positions,
                        __global float4* velocities,
                        __global const float4* accelerations,
                        const int n,
                        const float dt)
{
    int i = get_global_id(0);
    if (i >= n) return;
    
    float4 vel = velocities[i];
    float4 acc = accelerations[i];
    
    // Velocity Verlet integration
    vel += acc * dt;
    positions[i] += vel * dt;
    velocities[i] = vel;
}