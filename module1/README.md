# Module 1: Performance Measurement

## Instructions to run on CHPC
Get an interactive session on a CHPC node:

```
srun -M notchpeak --account=soc-gpu-np --partition=soc-gpu-np --nodes=1 -t 0:30:00  --gres=gpu:a100:1  --pty /bin/bash -l
```

Load these modules:

```
module load python cuda intel intel-mkl
```


## List of Benchmarks
Run each benchmark with commands below:
1. Interpreted Python

```
make run_python_dgemm_loops
```
2. Sequential Loops in C
```
make run_loops_dgemm
```
3. Parallel Loops in C
```
make run_loops_dgemm_parallel
```
4. Sequential MKL 
```
make run_mkl_dgemm
```
5. Parallel MKL
```
make run_mkl_dgemm_parallel
```
6. cuBLAS on GPU
```
make run_cublas_dgemm
```
7. Python NumPy
```
make run_python_dgemm_numpy
```
