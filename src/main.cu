#include <stdio.h>

void checkCudaError(cudaError_t error, const char *message) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s - %s\n", message, cudaGetErrorString(error));
        exit(-1);
    }
}

void printDeviceInfo() {
    int deviceCount;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);
    checkCudaError(error, "Failed to get device count");

    if (deviceCount == 0) {
        printf("No CUDA devices found!\n");
        exit(-1);
    }

    printf("Found %d CUDA device(s)\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        error = cudaGetDeviceProperties(&prop, i);
        checkCudaError(error, "Failed to get device properties");

        printf("\nDevice %d: %s\n", i, prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max thread dimensions: (%d, %d, %d)\n", 
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max grid dimensions: (%d, %d, %d)\n", 
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Global memory: %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    }
}

__global__ void addVectors(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    // Print device information
    printDeviceInfo();

    const int N = 1024;
    size_t size = N * sizeof(float);

    // Host arrays
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);

    // Initialize arrays
    for(int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }

    // Device arrays
    float *d_a, *d_b, *d_c;
    checkCudaError(cudaMalloc(&d_a, size), "Failed to allocate device memory for a");
    checkCudaError(cudaMalloc(&d_b, size), "Failed to allocate device memory for b");
    checkCudaError(cudaMalloc(&d_c, size), "Failed to allocate device memory for c");

    // Copy to device
    checkCudaError(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), 
                  "Failed to copy a to device");
    checkCudaError(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), 
                  "Failed to copy b to device");

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    addVectors<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);

    // Check for kernel launch errors
    checkCudaError(cudaGetLastError(), "Kernel launch failed");
    // Wait for kernel to finish and check for errors
    checkCudaError(cudaDeviceSynchronize(), "Kernel execution failed");

    // Copy result back
    checkCudaError(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), 
                  "Failed to copy result back to host");

    // Verify (print first 5 elements)
    printf("\nVerification (first 5 elements):\n");
    for(int i = 0; i < 5; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }

    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
