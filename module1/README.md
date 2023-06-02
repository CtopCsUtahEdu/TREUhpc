# Module 1: Performance Measurement

## List of Benchmarks
1. Interpreted Python
2. Sequential Loops in C
3. Parallel Loops in C
4. Sequential MKL 
5. Parallel MKL
6. cuBLAS on GPU
7. Python NumPy

## Instructions to run on CHPC
Get an interactive session on a CHPC node:

`srun -M notchpeak --account=soc-gpu-np --partition=soc-gpu-np --nodes=1 -t 0:30:00  --gres=gpu:a100:1  --pty /bin/bash -l`

Load these modules:

`module load python cuda intel intel-mkl`

Run each benchmark with `make`:

`make run_v1`

`make run_v2` 

...etc.
