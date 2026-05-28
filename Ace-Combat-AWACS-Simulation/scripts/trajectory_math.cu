#define _USE_MATH_DEFINES
#include <math.h>
#include <stdio.h>

// Extern C for ctypes
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT
#endif

extern "C" {
    EXPORT void calculate_interception(
        int num_targets,
        const float* x, const float* y, const float* z,
        const float* velocity, const float* heading,
        const int* aircraft_type_id,
        int num_bases,
        const float* base_x, const float* base_y, const float* base_z,
        float* tti, float* int_x, float* int_y, float* int_z,
        int* launch_base_idx,
        int* evasions
    );
}

// CUDA Kernel
__global__ void interception_kernel(
    int num_targets,
    const float* x, const float* y, const float* z,
    const float* velocity, const float* heading,
    const int* aircraft_type_id,
    int num_bases,
    const float* base_x, const float* base_y, const float* base_z,
    float* tti, float* int_x, float* int_y, float* int_z,
    int* launch_base_idx,
    int* evasions) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < num_targets) {
        float px = x[i];
        float py = y[i];
        float pz = z[i];
        float v = velocity[i];
        
        // Convert heading (degrees) to radians. 
        // 0 degrees is North (+y), 90 is East (+x)
        float h_rad = heading[i] * M_PI / 180.0f;
        float vx = v * sinf(h_rad);
        float vy = v * cosf(h_rad);
        float vz = 0.0f; // Target maintains altitude for simplicity
        
        // Interceptor speed depends on the target aircraft type (to simulate dynamic matching)
        // 0: MiG-29 -> F-15 interceptor (700 m/s)
        // 1: Su-27 -> F-14 interceptor (600 m/s)
        // 9: Su-47 Grabacr -> F-22A Mobius (850 m/s)
        // 10: Su-37 Gelb/Yellow -> F-15C Galm (750 m/s)
        // 11: Su-33 Strigon -> F-15E Garuda (720 m/s)
        // 12: Su-30SM Sol -> F-14D Wardog (680 m/s)
        // 13: Su-35 Ofnir -> F-16C Crow (650 m/s)
        // Other -> SAM Missile (1000 m/s)
        float Sm = 1000.0f;
        int target_type = aircraft_type_id[i];
        if (target_type == 0) {
            Sm = 700.0f; // Allied F-15 Eagle intercept speed (Mach 2+)
        } else if (target_type == 1) {
            Sm = 600.0f; // Allied F-14 Tomcat intercept speed (Mach 1.8+)
        } else if (target_type == 9) {
            Sm = 850.0f; // Allied F-22A Raptor intercept speed (Mach 2.5+)
        } else if (target_type == 10) {
            Sm = 750.0f; // Allied F-15C Eagle intercept speed (Mach 2.2+)
        } else if (target_type == 11) {
            Sm = 720.0f; // Allied F-15E Strike Eagle intercept speed (Mach 2.1)
        } else if (target_type == 12) {
            Sm = 680.0f; // Allied F-14D Super Tomcat intercept speed (Mach 2.0)
        } else if (target_type == 13) {
            Sm = 650.0f; // Allied F-16C Fighting Falcon intercept speed (Mach 1.9)
        }
        
        float best_tti = -1.0f;
        int best_base_idx = -1;
        float best_int_x = 0.0f;
        float best_int_y = 0.0f;
        float best_int_z = 0.0f;
        
        // Loop through all bases and solve quadratic intercept for each
        for (int b = 0; b < num_bases; ++b) {
            float bx = base_x[b];
            float by = base_y[b];
            float bz = base_z[b];
            
            float dx = px - bx;
            float dy = py - by;
            float dz = pz - bz;
            
            // Quadratic coefficients: a t^2 + b t + c = 0
            float a = (vx*vx + vy*vy + vz*vz) - (Sm*Sm);
            float b_coeff = 2.0f * (dx*vx + dy*vy + dz*vz);
            float c = (dx*dx + dy*dy + dz*dz);
            
            float discriminant = b_coeff*b_coeff - 4.0f*a*c;
            
            if (discriminant >= 0.0f) {
                float t1 = (-b_coeff + sqrtf(discriminant)) / (2.0f*a);
                float t2 = (-b_coeff - sqrtf(discriminant)) / (2.0f*a);
                
                float t = -1.0f;
                if (t1 > 0 && t2 > 0) t = fminf(t1, t2);
                else if (t1 > 0) t = t1;
                else if (t2 > 0) t = t2;
                
                if (t > 0.0f) {
                    if (best_tti < 0.0f || t < best_tti) {
                        best_tti = t;
                        best_base_idx = b;
                        best_int_x = px + vx * t;
                        best_int_y = py + vy * t;
                        best_int_z = pz + vz * t;
                    }
                }
            }
        }
        
        if (best_tti > 0.0f) {
            tti[i] = best_tti;
            launch_base_idx[i] = best_base_idx;
            int_x[i] = best_int_x;
            int_y[i] = best_int_y;
            int_z[i] = best_int_z;
        } else {
            tti[i] = -1.0f; // Cannot intercept
            launch_base_idx[i] = -1;
            int_x[i] = 0.0f;
            int_y[i] = 0.0f;
            int_z[i] = 0.0f;
            atomicAdd(evasions, 1);
        }
    }
}

// Host function wrapper
void calculate_interception(
    int num_targets,
    const float* x, const float* y, const float* z,
    const float* velocity, const float* heading,
    const int* aircraft_type_id,
    int num_bases,
    const float* base_x, const float* base_y, const float* base_z,
    float* tti, float* int_x, float* int_y, float* int_z,
    int* launch_base_idx,
    int* evasions) 
{
    float *d_x, *d_y, *d_z, *d_velocity, *d_heading;
    int *d_aircraft_type_id;
    float *d_base_x, *d_base_y, *d_base_z;
    float *d_tti, *d_int_x, *d_int_y, *d_int_z;
    int *d_launch_base_idx, *d_evasions;
    
    size_t size_float_targets = num_targets * sizeof(float);
    size_t size_int_targets = num_targets * sizeof(int);
    size_t size_float_bases = num_bases * sizeof(float);
    
    // Allocate device memory
    cudaMalloc((void**)&d_x, size_float_targets);
    cudaMalloc((void**)&d_y, size_float_targets);
    cudaMalloc((void**)&d_z, size_float_targets);
    cudaMalloc((void**)&d_velocity, size_float_targets);
    cudaMalloc((void**)&d_heading, size_float_targets);
    cudaMalloc((void**)&d_aircraft_type_id, size_int_targets);
    
    cudaMalloc((void**)&d_base_x, size_float_bases);
    cudaMalloc((void**)&d_base_y, size_float_bases);
    cudaMalloc((void**)&d_base_z, size_float_bases);
    
    cudaMalloc((void**)&d_tti, size_float_targets);
    cudaMalloc((void**)&d_int_x, size_float_targets);
    cudaMalloc((void**)&d_int_y, size_float_targets);
    cudaMalloc((void**)&d_int_z, size_float_targets);
    cudaMalloc((void**)&d_launch_base_idx, size_int_targets);
    cudaMalloc((void**)&d_evasions, sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_x, x, size_float_targets, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size_float_targets, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, size_float_targets, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity, velocity, size_float_targets, cudaMemcpyHostToDevice);
    cudaMemcpy(d_heading, heading, size_float_targets, cudaMemcpyHostToDevice);
    cudaMemcpy(d_aircraft_type_id, aircraft_type_id, size_int_targets, cudaMemcpyHostToDevice);
    
    cudaMemcpy(d_base_x, base_x, size_float_bases, cudaMemcpyHostToDevice);
    cudaMemcpy(d_base_y, base_y, size_float_bases, cudaMemcpyHostToDevice);
    cudaMemcpy(d_base_z, base_z, size_float_bases, cudaMemcpyHostToDevice);
    cudaMemcpy(d_evasions, evasions, sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_targets + threadsPerBlock - 1) / threadsPerBlock;
    
    interception_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        num_targets, d_x, d_y, d_z, d_velocity, d_heading, d_aircraft_type_id,
        num_bases, d_base_x, d_base_y, d_base_z,
        d_tti, d_int_x, d_int_y, d_int_z, d_launch_base_idx, d_evasions
    );
    cudaDeviceSynchronize();
    
    // Copy results back to host
    cudaMemcpy(tti, d_tti, size_float_targets, cudaMemcpyDeviceToHost);
    cudaMemcpy(int_x, d_int_x, size_float_targets, cudaMemcpyDeviceToHost);
    cudaMemcpy(int_y, d_int_y, size_float_targets, cudaMemcpyDeviceToHost);
    cudaMemcpy(int_z, d_int_z, size_float_targets, cudaMemcpyDeviceToHost);
    cudaMemcpy(launch_base_idx, d_launch_base_idx, size_int_targets, cudaMemcpyDeviceToHost);
    cudaMemcpy(evasions, d_evasions, sizeof(int), cudaMemcpyDeviceToHost);
    
    // Free device memory
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_velocity);
    cudaFree(d_heading);
    cudaFree(d_aircraft_type_id);
    cudaFree(d_base_x);
    cudaFree(d_base_y);
    cudaFree(d_base_z);
    cudaFree(d_tti);
    cudaFree(d_int_x);
    cudaFree(d_int_y);
    cudaFree(d_int_z);
    cudaFree(d_launch_base_idx);
    cudaFree(d_evasions);
}
