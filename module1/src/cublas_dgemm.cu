#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main() {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int numIterations = 10;

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

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Set alpha and beta values for the matrix multiplication
    double alpha = 1.0f;
    double beta = 0.0f;

    // Initialize variables for timing
    double totalTime = 0.0f;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int iter = 0; iter < numIterations; ++iter) {
        // Copy input matrices from host to device
        cudaMemcpy(d_A, h_A, m * k * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, k * n * sizeof(double), cudaMemcpyHostToDevice);

        // Start the timer
        cudaEventRecord(start);

        // Perform matrix multiplication using cuBLAS
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_B, n, d_A, k, &beta, d_C, n);

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
    printf (" Matrix multiplication using cuBLAS C=A*B, A(%i,%i) and B(%i,%i) \n"
            "  Average execution time = %.5f seconds == \n  Performance in GFLOPS = %.5f GFLOPS \n\n", m, k, k, n, averageTime / 1000, flops);

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;

    return 0;
}
