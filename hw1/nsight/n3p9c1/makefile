TARGETS = ../hw1
TEST=35
N=536869888
NODE_NUM=3
PROC_NUM=9
CPU_NUM=1
nsight_report:
	srun -N${NODE_NUM} -n${PROC_NUM} -c${CPU_NUM} /home/pp23/pp23s25/nsight-systems-2023.3.1/target-linux-x64/nsys \
	profile --mpi-impl=mpich --trace=mpi ./${TARGETS} ${N} testcases/${TEST}.in testcases/${TEST}.out
