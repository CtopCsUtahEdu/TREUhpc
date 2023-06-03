#include <iostream>
#include <cublas_v2.h>

#define N 1024  // Matrix size
#define M 1024  // Matrix size

int main() {
    double A[N*M], B[M*N], C[N*N];

    // Initialize matrices A and B (example values)
    // TODO: Modify this to match what Mahesh has for initialization.
    for (int i = 0; i < N*M; ++i) {
        A[i] = i;
        B[i] = i;
    }

    cublasHandle_t handle;
    cublasCreate(&handle);

    double *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, N * M * sizeof(double));
    cudaMalloc(&d_B, M * N * sizeof(double));
    cudaMalloc(&d_C, N * N * sizeof(double));

    cublasSetMatrix(N, M, sizeof(double), A, N, d_A, N);
    cublasSetMatrix(M, N, sizeof(double), B, M, d_B, M);

    const double alpha = 1.0;
    const double beta = 0.0;

    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, M, &alpha, d_A, N, d_B, M, &beta, d_C, N);

    cublasGetMatrix(N, N, sizeof(double), d_C, N, C, N);

/*    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }*/

    cublasDestroy(handle);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
