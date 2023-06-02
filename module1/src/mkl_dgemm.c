/**********************************************************************
*  This benchmark measures performance of Intel MKL function dgemm
*  computing real matrix C = alpha * A * B + beta * C, where A, B, and C 
*  are matrices and alpha and beta are double precision scalars
***********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include "mkl.h"

#define LOOP_COUNT 10

double get_time();

int main()
{
    mkl_set_num_threads_local(1);
    double *A, *B, *C;
    int m, n, p, i, j, k, r;
    double alpha, beta;
    double s_initial, s_elapsed;

    // Matrix dimensions
    m = 1024, p = 1024, n = 1024;
    
    alpha = 1.0; beta = 1.0;

    // Allocating memory for the matrices  aligned on the 64-byte boundary 
    A = (double *)mkl_malloc( m*p*sizeof( double ), 64 );
    B = (double *)mkl_malloc( p*n*sizeof( double ), 64 );
    C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );

    /* A = (double *)malloc( m*p*sizeof( double ));
     B = (double *)malloc( p*n*sizeof( double ));
     C = (double *)malloc( m*n*sizeof( double )); */
    if (A == NULL || B == NULL || C == NULL) {
        printf( "\n ERROR: Can't allocate memory for matrices. Aborting... \n\n");
        mkl_free(A);
        mkl_free(B);
        mkl_free(C);
        return 1;
    }
  
    // Initializing data for gemm
    for (i = 0; i < (m*p); i++) {
        A[i] = (double)((i+1)*1.213);
    }

    for (i = 0; i < (p*n); i++) {
        B[i] = (double)((-i-1)*1.543);
    }

    for (i = 0; i < (m*n); i++) {
        C[i] = 0.0;
    }

    mkl_set_num_threads(1);

    // First run of the dgemm kernel to get stable run time measurements
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                m, n, p, alpha, A, p, B, n, beta, C, n);

    s_initial = get_time();
    for (r = 0; r < LOOP_COUNT; r++) {
        mkl_set_num_threads(1);
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, 
                    m, n, p, alpha, A, p, B, n, beta, C, n);
    }
    s_elapsed = (get_time() - s_initial) / LOOP_COUNT;
    double flops = (2.0e-9*m*n*p)/s_elapsed;

    printf (" Matrix multiplication using MKL dgemm C=A*B, A(%i,%i) and B(%i,%i) \n"
            "  Execution time = %.5f seconds == \n  Performance in GFLOPS = %.5f GFLOPS \n\n", m, p, p, n, s_elapsed, flops);
    
    //Granularity Check
    if (s_elapsed < 0.9/LOOP_COUNT) {
        s_elapsed=1.0/LOOP_COUNT/s_elapsed;
        i=(int)(s_elapsed*LOOP_COUNT)+1;
        printf(" Warning: Define LOOP_COUNT for this benchmark as %i \n" 
               " to have total execution time about 1 second for reliability \n"
               " of measurements\n\n", i);
    }

  

    //Correctness Check
  /*  double *exp_A = (double *)mkl_malloc( m*p*sizeof( double ), 64 );
    double *exp_B = (double *)mkl_malloc( p*n*sizeof( double ), 64 );
    double *exp_C = (double *)mkl_malloc( m*n*sizeof( double ), 64 );

    double epsilon = 1.0e-13;

    int err = 0;    
  
    for (i = 0; i < (m*p); i++) {
        exp_A[i] = (double)((i+1)*1.213);
    }

    for (i = 0; i < (p*n); i++) {
        exp_B[i] = (double)((-i-1)*1.543);
    }

    for (i = 0; i < (m*n); i++) {
        exp_C[i] = 0.0;
    }
   
    for (r = 0; r < LOOP_COUNT+1; r++) {
      for (i = 0; i < m; i++) {
         for (j = 0; j < n; j++) {
            for (k = 0; k < p; k++) {
               exp_C[n*i+j] =  beta*exp_C[n*i+j] + alpha*exp_A[p*i+k] * exp_B[n*k+j];
            }
         }
       }
    }

    for (i = 0; i < m; i++) {
      for (j = 0; j < n; j++) {
           //printf("%.5f & %.5f   ", exp_C[i*n+j], C[i*n+j]); 
           if (abs(exp_C[i*n+j] - C[i*n+j]) >= epsilon)
		err++;
      } printf("\n");
    }

   if (err > 0) {
       printf("Correctness Test: Failed \n");
       printf("Num_err = %d\n", err);
   }
   else if(err == 0)
      printf("Correctness Test: PASSED \n");

*/
   mkl_free(A);
   mkl_free(B);
   mkl_free(C);
   //mkl_free(exp_A);
   //mkl_free(exp_B);
   //mkl_free(exp_C);
 
   return 0;   
}
double get_time(){
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ( (double) tp.tv_sec + (double) tp.tv_usec * 1.0e-6 );
}
