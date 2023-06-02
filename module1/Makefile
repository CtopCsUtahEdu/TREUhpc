
CC := icc
SRCDIR := src
BUILDDIR := bin
LIBFLAGS := -qopenmp -mkl -static-intel

all: $(BUILDDIR)/mkl_dgemm $(BUILDDIR)/loops_dgemm $(BUILDDIR)/mkl_dgemm_parallel $(BUILDDIR)/loops_dgemm_parallel

$(BUILDDIR)/%: $(BUILDDIR)/%.o
	$(CC) $< $(LIBFLAGS) -o $@

$(BUILDDIR)/%.o: $(SRCDIR)/%.c
	@mkdir -p $(BUILDDIR)
	$(CC) -c $< -o $@

run_v3: $(BUILDDIR)/mkl_dgemm
	./$(BUILDDIR)/mkl_dgemm

run_v2: $(BUILDDIR)/loops_dgemm
	./$(BUILDDIR)/loops_dgemm

run_v5: $(BUILDDIR)/mkl_dgemm_parallel
	./$(BUILDDIR)/mkl_dgemm_parallel

run_v4: $(BUILDDIR)/loops_dgemm_parallel
	./$(BUILDDIR)/loops_dgemm_parallel

clean:
	@echo " Cleaning..."
	@rm -fr $(BUILDDIR) 2>/dev/null || true

.PHONY: clean
.PRECIOUS: $(BUILDDIR)/%.o
