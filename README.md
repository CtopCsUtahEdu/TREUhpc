# HPC Modules for Utah KSoC TREU 2023
Instructor: Prof. Mary Hall (mhall@cs.utah.edu)

## Module 1: Performance Measurement
Performance comparison of matrix multiplication kernel on CPUs and GPUs. 




____
How can you know details about your CPU on your device ?
`lscpu`
How to know what GPU does your machine have ?
incase of Nvidia GPU
`nvidia-smi`
todo exercise 
try for AMD machine.


Clone this repository
How ?
`git clone https://github.com/CtopCsUtahEdu/TREUhpc.git`

Go into the repository
`cd TREUhpc/module1`
look around 

```
1. There are 2 files and 1 folder 
Try opening the `README.md` file and try to read it and understand what it has, this file will have usually instructions about the current project, it's a good idea in any large project to read them to get an understanding of how to build the project, and more details about the structure of the files in project, it tell you more about which folder contains which related files.

A Makefile 

```
A Makefile is a text file that contains a set of instructions or rules used by the make utility to build and manage projects. It is commonly used in software development to automate the compilation and linking of source code into executable binaries or libraries.

A simple example of how a Makefile looks is as below
`
program: main.c utils.c
    gcc -o program main.c utils.c

`
What this does is when you type a single command `make program` this will internally call the compiler `gcc` and pass 2 input files to it `main.c` and `utils.c` and generate the output executable named `program` (Note: Usually you might be familiar with `a.out` you can control the output name with `-o` flag.)

Makefiles offer a powerful and flexible way to automate software builds and manage project dependencies. They are widely used in various programming languages and development environments to streamline the build process and improve productivity.
Note: CMAKE, BAZEL are the recent and even more powerful build systems in production level softwares used today.

More reading:
1. List of build automation software
https://en.wikipedia.org/wiki/List_of_build_automation_software
2. Get an idea of various types of build systems, you don't need to stress to know all of them just get to know how and what they try to achieve
https://julienjorge.medium.com/an-overview-of-build-systems-mostly-for-c-projects-ac9931494444

``

Next step :
1. Go into `src` repository,
`cd src/`
2. Try opening files and look around and try checking the codes, again if you don't understand the specifics of the source code that should be OK, just get the basic idea of what they try to do
