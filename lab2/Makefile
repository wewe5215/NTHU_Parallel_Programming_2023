CC = gcc
CXX = g++
CFLAGS = -lm -O3
lab2_pthread: CFLAGS += -pthread
lab2_omp: CFLAGS += -fopenmp
lab2_hybrid: CC = mpicc
lab2_hybrid: CXX = mpicxx
lab2_hybrid: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = lab2_pthread lab2_omp lab2_hybrid
R=5
K=21
.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
run_pthread:
	srun -c4 -n1 ./lab2_pthread ${R} ${K} > pthread_output.txt
run_omp:
	srun -c4 -n1 ./lab2_omp ${R} ${K} > omp_output.txt
run_hybrid:
	srun -c3 -n2 ./hello_hybrid