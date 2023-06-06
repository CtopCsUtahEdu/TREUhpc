#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <mkl.h>

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void matrixMul(double *A, double *B, double *C, int m, int k, int n) {
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                m, n, k, 1.0, A, k, B, n, 0.0, C, n);
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

    // Create variables for timing
    double startTime, endTime;
    double totalTime = 0.0;

    for (int iter = 0; iter < numIterations; ++iter) {
        // Clear the result matrix
        memset(C, 0, m * n * sizeof(double));

        // Start the timer
        startTime = get_time();

        matrixMul(A, B, C, m, k, n);

        // Stop the timer
        endTime = get_time();

        // Calculate elapsed time
        double elapsedTime = endTime - startTime;
        totalTime += elapsedTime;
    }

    // Calculate average elapsed time
    double averageTime = totalTime / numIterations;

    // Calculate number of floating-point operations (FLOPs)
    double flops = (2.0e-9 * m * n * k) / averageTime;

    // Print the average elapsed time and FLOPs
    printf("Matrix multiplication using MKL: A(%d, %d) * B(%d, %d) = C(%d, %d)\n", m, k, k, n, m, n);
    printf("Average execution time: %.5f seconds\n", averageTime);
    printf("Performance in GFLOPs: %.5f GFLOPs\n", flops);

    // Cleanup
    free(A);
    free(B);
    free(C);

    return 0;
}
