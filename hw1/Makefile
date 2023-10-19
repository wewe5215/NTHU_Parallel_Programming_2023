CC = mpicc
CXX = mpicxx
CXXFLAGS =-std=c++17 -O3 -lm
#CXXFLAGS = -std=c++17 -O3 -lm -L/opt/ipm_mpi/lib -lipm
TARGETS = hw1
NODE_NUM=1
PROC_NUM=12
TEST=35
.PHONY: all
all: 
	$(CXX) $(CXXFLAGS) -o $(TARGETS) $(TARGETS).cc
.PHONY: clean
clean:
	rm -f $(TARGETS)

run :
	srun -N${NODE_NUM} -n${PROC_NUM} ./${TARGETS} 536869888 testcases/${TEST}.in testcases/${TEST}.out
nsight_report:
	srun -N${NODE_NUM} -n${PROC_NUM} /home/pp23/pp23s25/nsight-systems-2023.3.1/target-linux-x64/nsys \
	profile --trace=mpi ./${TARGETS} 50 testcases/${TEST}.in testcases/${TEST}.out
gen_report:
	IPM_REPORT=full IPM_REPORT_MEM=yes IPM_LOG=full LD_PRELOAD=/opt/ipm/lib/libipm.so \
	srun -N${NODE_NUM} -n${PROC_NUM} ./${TARGETS} 536869888 testcases/${TEST}.in testcases/${TEST}.out