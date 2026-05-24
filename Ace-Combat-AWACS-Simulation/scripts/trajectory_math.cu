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
        float* tti, float* int_x, float* int_y, float* int_z,
        int* evasions
    );
}

// CUDA Kernel
__global__ void interception_kernel(
    int num_targets,
    const float* x, const float* y, const float* z,
    const float* velocity, const float* heading,
    float* tti, float* int_x, float* int_y, float* int_z,
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
        
        float Sm = 1000.0f; // SAM Speed (1000 m/s)
        
        // Quadratic coefficients: a t^2 + b t + c = 0
        float a = (vx*vx + vy*vy + vz*vz) - (Sm*Sm);
        float b = 2.0f * (px*vx + py*vy + pz*vz);
        float c = (px*px + py*py + pz*pz);
        
        float discriminant = b*b - 4.0f*a*c;
        
        if (discriminant >= 0.0f) {
            float t1 = (-b + sqrtf(discriminant)) / (2.0f*a);
            float t2 = (-b - sqrtf(discriminant)) / (2.0f*a);
            
            float t = -1.0f;
            if (t1 > 0 && t2 > 0) t = fminf(t1, t2);
            else if (t1 > 0) t = t1;
            else if (t2 > 0) t = t2;
            
            if (t > 0.0f) {
                tti[i] = t;
                int_x[i] = px + vx * t;
                int_y[i] = py + vy * t;
                int_z[i] = pz + vz * t;
            } else {
                tti[i] = -1.0f; // Cannot intercept
                int_x[i] = 0.0f;
                int_y[i] = 0.0f;
                int_z[i] = 0.0f;
                atomicAdd(evasions, 1);
            }
        } else {
            tti[i] = -1.0f;
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
    float* tti, float* int_x, float* int_y, float* int_z,
    int* evasions) 
{
    float *d_x, *d_y, *d_z, *d_velocity, *d_heading;
    float *d_tti, *d_int_x, *d_int_y, *d_int_z;
    int *d_evasions;
    
    size_t size = num_targets * sizeof(float);
    
    cudaMalloc((void**)&d_x, size);
    cudaMalloc((void**)&d_y, size);
    cudaMalloc((void**)&d_z, size);
    cudaMalloc((void**)&d_velocity, size);
    cudaMalloc((void**)&d_heading, size);
    cudaMalloc((void**)&d_tti, size);
    cudaMalloc((void**)&d_int_x, size);
    cudaMalloc((void**)&d_int_y, size);
    cudaMalloc((void**)&d_int_z, size);
    cudaMalloc((void**)&d_evasions, sizeof(int));
    
    cudaMemcpy(d_x, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_z, z, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_velocity, velocity, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_heading, heading, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_evasions, evasions, sizeof(int), cudaMemcpyHostToDevice);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_targets + threadsPerBlock - 1) / threadsPerBlock;
    
    interception_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        num_targets, d_x, d_y, d_z, d_velocity, d_heading,
        d_tti, d_int_x, d_int_y, d_int_z, d_evasions
    );
    cudaDeviceSynchronize();
    
    cudaMemcpy(tti, d_tti, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(int_x, d_int_x, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(int_y, d_int_y, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(int_z, d_int_z, size, cudaMemcpyDeviceToHost);
    cudaMemcpy(evasions, d_evasions, sizeof(int), cudaMemcpyDeviceToHost);
    
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    cudaFree(d_velocity);
    cudaFree(d_heading);
    cudaFree(d_tti);
    cudaFree(d_int_x);
    cudaFree(d_int_y);
    cudaFree(d_int_z);
    cudaFree(d_evasions);
}
