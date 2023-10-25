CC = mpicc
CXX = mpicxx
CXXFLAGS =-std=c++17 -O3 -lm
TARGETS = hw1
TEST=35
N=536869888
NODE_NUM=1
PROC_NUM=8
CPU_NUM=1
.PHONY: all
all: 
	$(CXX) $(CXXFLAGS) -o $(TARGETS) $(TARGETS).cc
.PHONY: clean
clean:
	rm -f $(TARGETS)

run:
	srun -N${NODE_NUM} -n${PROC_NUM} -c${CPU_NUM} ./${TARGETS} ${N} testcases/${TEST}.in testcases/${TEST}.out