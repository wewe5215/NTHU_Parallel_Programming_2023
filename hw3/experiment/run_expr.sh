#!/bin/bash
CC=gcc
CXX=g++
CXXFLAGS="-O3 -pthread"
CFLAGS="-O3 -lm -pthread"
NVFLAGS="-std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -use_fast_math"
LDFLAGS="-lm"
run_command() {
    FILE_NAME=$1
    srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics ${FILE_NAME} ./hw3-2 /home/pp23/share/hw3-2/cases/p24k1 p24k1.out &> experiment_result/${FILE_NAME}.txt
}
run_BF() {
    FILE_NAME=$1
    nvcc ${NVFLAGS} ${LDFLAGS} -o ${FILE_NAME} ${FILE_NAME}.cu
    srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics inst_integer ./${FILE_NAME} /home/pp23/share/hw3-2/cases/c21.1 ${FILE_NAME}_c21.1.out &> experiment_result/inst_integer_${FILE_NAME}.txt
    srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics gld_throughput ./${FILE_NAME} /home/pp23/share/hw3-2/cases/c21.1 ${FILE_NAME}_c21.1.out &> experiment_result/gld_throughput_${FILE_NAME}.txt
    srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics gst_throughput ./${FILE_NAME} /home/pp23/share/hw3-2/cases/c21.1 ${FILE_NAME}_c21.1.out &> experiment_result/gst_throughput_${FILE_NAME}.txt
    srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics shared_load_throughput ./${FILE_NAME} /home/pp23/share/hw3-2/cases/c21.1 ${FILE_NAME}_c21.1.out &> experiment_result/shared_load_throughput_${FILE_NAME}.txt
    srun -p prof -N1 -n1 --gres=gpu:1 nvprof --metrics shared_store_throughput ./${FILE_NAME} /home/pp23/share/hw3-2/cases/c21.1 ${FILE_NAME}_c21.1.out &> experiment_result/shared_store_throughput_${FILE_NAME}.txt
}

# run_command sm_efficiency
# run_command shared_load_throughput
# run_command shared_store_throughput
# run_command gld_throughput
# run_command gst_throughput
# run_BF bf_8
# run_BF bf_16
# run_BF bf_32
# run_BF bf_64 --> done
echo "All commands completed."