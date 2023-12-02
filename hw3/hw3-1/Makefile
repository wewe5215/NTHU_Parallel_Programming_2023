CC = gcc
CXX = g++
CXXFLAGS = -O3 -pthread
CFLAGS = -O3 -lm -pthread
TARGETS = hw3-1

.PHONY: all
all: $(TARGETS)

.PHONY: clean
clean:
	rm -f $(TARGETS)
run:
	srun -N1 -n1 -c1 ./hw3-1 sample/cases/c01.1 c01.1.out