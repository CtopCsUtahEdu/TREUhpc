/*******************************************************************************
*  Benchmark for parallel matrix multiplication with a triple nested loop 
********************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "mkl.h"
#include <sys/time.h>

#define LOOP_COUNT 1

double get_time();

int main()
{
    double *A, *B, *C;
    int m, n, p, i, j, k, r, max_threads;
    double alpha, beta;
    double sum;
    double s_initial, s_elapsed;

    //Set matrix dimensions
    m = 1024, p = 1024, n = 1024;
    
    alpha = 1.0; beta = 0.0;
    
    A = (double *)mkl_malloc( m*p*sizeof( double ), 64 );
    B = (double *)mkl_malloc( p*n*sizeof( double ), 64 );
    C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );
    if (A == NULL || B == NULL || C == NULL) {
        printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return 1;
    }

    //Intializing matrix data
    for (i = 0; i < (m*p); i++) {
        A[i] = (double)(i+1);
    }

    for (i = 0; i < (p*n); i++) {
        B[i] = (double)(-i-1);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }

    //Find max number of threads
    max_threads = omp_get_max_threads();

    //Set up OpenMP to use max_threads number of threads
    omp_set_num_threads(max_threads);

    //First run of matrix product to get stable run time measurements
    #pragma omp parallel for
    for (i = 0; i < m; i++) {
        for (j = 0; j < n; j++) {
            sum = 0.0;
            for (k = 0; k < p; k++)
                sum += A[p*i+k] * B[n*k+j];
            C[n*i+j] = sum;
        }
    }

    //Measuring the performance of matrix product
    s_initial = get_time();
    #pragma omp parallel for
    for (r = 0; r < LOOP_COUNT; r++) {
        for (i = 0; i < m; i++) {
            for (j = 0; j < n; j++) {
                sum = 0.0;
                for (k = 0; k < p; k++)
                    sum += A[p*i+k] * B[n*k+j];
                C[n*i+j] = sum;
            }
        }
    }
    
    s_elapsed = (get_time() - s_initial) / LOOP_COUNT;
   
    double flops = (2.0e-9*m*n*p)/s_elapsed;
 
    printf (" == Matrix multiplication with triple nested loop in parallel using %d threads: C=A*B, A(%i,%i) and B(%i,%i)== \n"
            " == Execution Time = %.5f seconds == \n == Performance in FLOPS = %.5f GFLOPS == \n\n", max_threads, m, n, n, p, s_elapsed, flops);
    
    mkl_free(A);
    mkl_free(B);
    mkl_free(C);
    
    if (s_elapsed < 0.9/LOOP_COUNT) {
        s_elapsed=1.0/LOOP_COUNT/s_elapsed;
        i=(int)(s_elapsed*LOOP_COUNT)+1;
        printf(" It is highly recommended to define LOOP_COUNT for this benchmark \n"
               " as %i to have total execution time about 1 second for reliability \n"
               " of measurements\n\n", i);
    }
    
    return 0;
}

double get_time(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.0e-6 );
}	

