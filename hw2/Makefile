CC = clang
CXX = clang++
LDLIBS = -lpng
CFLAGS = -lm -O3
hw2a: CFLAGS += -pthread -ffast-math
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp -fno-math-errno -ffinite-math-only -freciprocal-math \
-fno-signed-zeros -fno-trapping-math -fno-rounding-math -ffp-contract=fast
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
run_pthread:
	srun -n12 -N1 -c1 ./hw2a out.png 174170376 -0.7894722222222222 -0.7825277777777778 0.145046875 0.148953125 2549 1439
run_hybrid:
	srun -n3 -N2 -c1 ./hw2b out_hy.png 1476 -0.5 0.5 -0.5 0.5 491 935 > output2.txt