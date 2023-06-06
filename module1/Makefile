
CC := icc
CFLAGS = -qopenmp -mkl 
NVCC := nvcc
NVCCFLAGS := -arch=sm_80 -O3
SRCDIR := src
BUILDDIR := bin
LIBFLAGS := -qopenmp -mkl -static-intel -L$(CUDA_HOME)/lib64 -lcudart -lcurand -lcublas

all: $(BUILDDIR)/mkl_dgemm $(BUILDDIR)/loops_dgemm $(BUILDDIR)/mkl_dgemm_parallel $(BUILDDIR)/loops_dgemm_parallel $(BUILDDIR)/cuda_dgemm $(BUILDDIR)/cublas_dgemm

$(BUILDDIR)/%: $(BUILDDIR)/%.o
	$(CC) $< $(LIBFLAGS) -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.cu
	@mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

run_mkl_dgemm: $(BUILDDIR)/mkl_dgemm
	./$(BUILDDIR)/mkl_dgemm

run_loops_dgemm: $(BUILDDIR)/loops_dgemm
	./$(BUILDDIR)/loops_dgemm

run_mkl_dgemm_parallel: $(BUILDDIR)/mkl_dgemm_parallel
	./$(BUILDDIR)/mkl_dgemm_parallel

run_loops_dgemm_parallel: $(BUILDDIR)/loops_dgemm_parallel
	./$(BUILDDIR)/loops_dgemm_parallel

run_cuda_dgemm: $(BUILDDIR)/cuda_dgemm
	./$(BUILDDIR)/cuda_dgemm

run_cublas_dgemm: $(BUILDDIR)/cublas_dgemm
	./$(BUILDDIR)/cublas_dgemm

clean:
	@echo " Cleaning..."
	@rm -fr $(BUILDDIR) 2>/dev/null || true

.PHONY: clean
.PRECIOUS: $(BUILDDIR)/%.o
