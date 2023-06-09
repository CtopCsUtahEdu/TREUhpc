# Module 1: Performance Measurement

**Note**: All of these commands work on Linx machine.

## Instructions to run on CHPC
Get an interactive session on a CHPC node:

```
srun -M kingspeak --cpus-per-task=28 --account=soc-gpu-kp --partition=soc-gpu-kp --nodes=1 -t 0:30:00  --gres=gpu:p100:1  --pty /bin/bash -l
```

Load these modules:

```
module load python cuda intel intel-mkl
```

Clone a github repository:

```
git clone https://github.com/CtopCsUtahEdu/TREUhpc.git
```

go into the repository

```
cd TREUhpc/module1
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

## Extra stuff

### Find hardware details
**How can you know details about your CPU on your device ?**

`lscpu`

**How to know which GPU does your machine have ?**

Specifically for Nvidia GPU

`nvidia-smi`

Query cpu and gpu information on your machine using commands mentioned above, why ? 
knowing which hardware configuration your machine has helps to reason you about the possible performance gains or performance degrades you might have, there can be cases when 2 exactly same codes can perform differently on 2 different machines having different hardware configurations.


### look around 

1. There are 2 files 
- a. README.md
- b. Makefile

**README.md**

Try opening the `README.md` file and try to read it and understand what it has, this file will have usually instructions about the current project, it's a good idea in any large project to read them to get an understanding of how to build the project, and more details about the structure of the files in project, it tell you more about which folder contains which related files.

**Makefile**

A Makefile is a text file that contains a set of instructions or rules used by the make utility to build and manage projects. It is commonly used in software development to automate the compilation and linking of source code into executable binaries or libraries.
A simple demo example of how a `Makefile` looks is as below
```
program: main.c utils.c
    gcc -o program main.c utils.c
```
When you type a single command `make program` this will internally call the compiler `gcc` and pass 2 input files to it `main.c` and `utils.c` and generate the output executable named `program` (Note: Usually you might be familiar with `a.out` you can control the output name with `-o` flag.)
Makefiles offer a powerful and flexible way to automate software builds and manage project dependencies. They are widely used in various programming languages and development environments to streamline the build process and improve productivity.
Note: CMAKE, BAZEL are the recent and even more powerful build systems in production level softwares used today.

**More reading to get some context :**
1. List of build automation software
https://en.wikipedia.org/wiki/List_of_build_automation_software
2. Get an idea of various types of build systems, you don't need to stress to know all of them just get to know how and what they try to achieve
https://julienjorge.medium.com/an-overview-of-build-systems-mostly-for-c-projects-ac9931494444

**A few more tasks you can do:**
1. Go into `src` repository,
`cd src/`
2. Try opening files and look around and try checking the codes, again if you don't understand the specifics of the source code that should be OK, just get the basic idea of what they try to do and how they differ.

