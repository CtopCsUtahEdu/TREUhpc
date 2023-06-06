#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 16

__global__ void matrixMul(double *A, double *B, double *C, int m, int k_total, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < m && col < n) {
        double sum = 0.0f;

        for (int k = 0; k < k_total; ++k) {
            sum += A[row * k_total + k] * B[k * n + col];
        }

        C[row * n + col] = sum;
    }
}

int main() {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int numIterations = 100;

    // Allocate memory on the host
    double *h_A = new double[m * k];
    double *h_B = new double[k * n];
    double *h_C = new double[m * n];

    // Initialize input matrices
    for (int i = 0; i < m * k; ++i) {
        h_A[i] = i + 1;
    }

    for (int i = 0; i < k * n; ++i) {
        h_B[i] = - i - 1;
    }

    // Allocate memory on the device
    double *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, m * k * sizeof(double));
    cudaMalloc((void**)&d_B, k * n * sizeof(double));
    cudaMalloc((void**)&d_C, m * n * sizeof(double));

    // Create CUDA events for timing measurement
    double totalTime = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < numIterations; ++iter) {
        // Copy input matrices from host to device
        cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

        // Set grid and block sizes
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((m + blockSize.x - 1) / blockSize.x, (n + blockSize.y - 1) / blockSize.y);

        // Start the timer
        cudaEventRecord(start);

        // Launch the matrix multiplication kernel
        matrixMul<<<gridSize, blockSize>>>(d_A, d_B, d_C, m, k, n);

        // Stop the timer
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate elapsed time
        float milliseconds = 0.0f;
        cudaEventElapsedTime(&milliseconds, start, stop);
        totalTime += milliseconds;

        // Copy result matrix from device to host
        cudaMemcpy(h_C, d_C, m * n * sizeof(double), cudaMemcpyDeviceToHost);
    }

    // Calculate average elapsed time
    double averageTime = totalTime / numIterations;

    // Print the average elapsed time
    double flops = (2.0e-9 * m * n * k) / (averageTime / 1000);
    printf (" Matrix multiplication using CUDA C=A*B, A(%i,%i) and B(%i,%i) \n"
            "  Average execution time = %.5f seconds == \n  Performance in GFLOPS = %.5f GFLOPS \n\n", m, k, k, n, averageTime / 1000, flops);

    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
