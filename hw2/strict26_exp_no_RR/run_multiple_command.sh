#!/bin/bash

# Compile hw2b_exp_nrr
CC=mpicc
CXX=mpicxx
LDLIBS=-lpng
CFLAGS="-lm -O3 -fopenmp -fno-math-errno -ffinite-math-only -freciprocal-math \
-fno-signed-zeros -fno-trapping-math -fno-rounding-math -ffp-contract=fast"

$CXX $CFLAGS -o hw2b_exp_nrr hw2b_exp_nrr.cc $LDLIBS

# Function to run the commands
run_commands() {
    NODE_NUM=$1
    PROC_NUM=$2
    CPU_NUM=$3

    # Run command
    srun -N${NODE_NUM} -n${PROC_NUM} -c${CPU_NUM} /home/pp23/pp23s25/nsight-systems-2023.3.1/target-linux-x64/nsys \
	    profile --mpi-impl=mpich --trace=mpi ./hw2b_exp_nrr out_hybrid_exp_nrr.png 10000 -0.19985997516420825 -0.19673850548118335 -1.0994739550641088 -1.1010040371755099 7680 4320

    # Additional commands
    mkdir n${NODE_NUM}p${PROC_NUM}c${CPU_NUM}
    mv report*.nsys-rep n${NODE_NUM}p${PROC_NUM}c${CPU_NUM}
}

# Run commands with different values
run_commands 1 1 1
run_commands 1 2 1
run_commands 1 4 1
run_commands 1 8 1
run_commands 1 12 1
run_commands 2 2 1
run_commands 2 4 1
run_commands 2 8 1
run_commands 2 16 1
run_commands 2 24 1
# Add more lines for additional configurations

echo "All commands completed."

