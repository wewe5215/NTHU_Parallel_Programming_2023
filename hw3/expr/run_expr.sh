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
run_optim(){
    FILE_NAME=$1
    nvcc ${NVFLAGS} ${LDFLAGS} -o ${FILE_NAME} ${FILE_NAME}.cu
    srun -N1 -n1 --gres=gpu:1 ./${FILE_NAME} /home/pp23/share/hw3-2/cases/p24k1 p24k1.out > ${FILE_NAME}_elapsed.txt
}
run_Time_Dist(){
    FILE_NAME=$1
    nvcc ${NVFLAGS} ${LDFLAGS} -o ${FILE_NAME} ${FILE_NAME}.cu
    srun -N1 -n1 --gres=gpu:1 ./${FILE_NAME} /home/pp23/share/hw3-2/cases/p15k1 p15k1.out > p15_final_elapsed.txt
    srun -N1 -n1 --gres=gpu:1 ./${FILE_NAME} /home/pp23/share/hw3-2/cases/p15k1 p20k1.out > p20_final_elapsed.txt
    srun -N1 -n1 --gres=gpu:1 ./${FILE_NAME} /home/pp23/share/hw3-2/cases/p25k1 p15k1.out > p25_final_elapsed.txt
    srun -N1 -n1 --gres=gpu:1 ./${FILE_NAME} /home/pp23/share/hw3-2/cases/p30k1 p30k1.out > p30_final_elapsed.txt
}

run_multi(){
    FILE_NAME=$1
    nvcc ${NVFLAGS} -Xcompiler -fopenmp ${LDFLAGS} -o ${FILE_NAME} ${FILE_NAME}.cu
    srun -N1 -n1 -c2 --gres=gpu:2 ./${FILE_NAME} /home/pp23/share/hw3-2/cases/p15k1 mp15k1.out > multi_p15_final_elapsed.txt
    srun -N1 -n1 -c2 --gres=gpu:2 ./${FILE_NAME} /home/pp23/share/hw3-2/cases/p15k1 mp20k1.out > multi_p20_final_elapsed.txt
    srun -N1 -n1 -c2 --gres=gpu:2 ./${FILE_NAME} /home/pp23/share/hw3-2/cases/p25k1 mp15k1.out > multi_p25_final_elapsed.txt
    srun -N1 -n1 -c2 --gres=gpu:2 ./${FILE_NAME} /home/pp23/share/hw3-2/cases/p30k1 mp30k1.out > multi_p30_final_elapsed.txt
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
# run_optim original
# run_optim shared_mem_TLE
# run_optim shared_mem_pass
# run_optim unroll
# run_optim optimize_phase3
run_Time_Dist final_version
run_multi final_version_2Gpu
echo "All commands completed."