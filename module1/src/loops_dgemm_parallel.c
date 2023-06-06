#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <sys/time.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void matrixMul(double *A, double *B, double *C, int m, int k, int n) {
    #pragma omp parallel for
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += A[i * k + l] * B[l * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

int main() {
    int m = 1024;
    int n = 1024;
    int k = 1024;
    int numIterations = 10;

    // Allocate memory for matrices
    double *A = (double*)malloc(m * k * sizeof(double));
    double *B = (double*)malloc(k * n * sizeof(double));
    double *C = (double*)malloc(m * n * sizeof(double));

    // Initialize input matrices
    for (int i = 0; i < m * k; ++i) {
        A[i] = i + 1;
    }

    for (int i = 0; i < k * n; ++i) {
        B[i] = -i - 1;
    }

    // Set the number of threads for OpenMP
    int numThreads = omp_get_max_threads();
    omp_set_num_threads(numThreads);

    // Create variables for timing
    double startTime, endTime;
    double totalTime = 0.0;

    for (int iter = 0; iter < numIterations; ++iter) {
        // Clear the result matrix
        memset(C, 0, m * n * sizeof(double));

        // Start the timer
        startTime = get_time();

        // Perform matrix multiplication using loops and OpenMP
        matrixMul(A, B, C, m, k, n);

        // Stop the timer
        endTime = get_time();

        // Calculate elapsed time
        double elapsedTime = endTime - startTime;
        totalTime += elapsedTime;
    }

    // Calculate average elapsed time
    double averageTime = totalTime / numIterations;

    // Calculate number of doubleing point operations (FLOPs)
    double flops = (2.0e-9 * m * n * k) / averageTime;

    // Print the average elapsed time, number of threads, and FLOPs
    printf("Matrix multiplication using loops and OpenMP: A(%d, %d) * B(%d, %d) = C(%d, %d)\n", m, k, k, n, m, n);
    printf("Average execution time: %.5f seconds\n", averageTime);
    printf("Number of threads: %d\n", numThreads);
    printf("Performance in GFLOPs: %.5f GFLOPs\n", flops);

    // Cleanup
    free(A);
    free(B);
    free(C);

    return 0;
}
