import re
with open('n1p8c1_1000000.txt', 'r') as file:
    lines = file.readlines()
tasks = []
apptimes = []
mpitimes = []
precent = []
for line in lines:
    line = line.strip().split('\n')[0]
    values = re.split(r'\s+', line.strip())
    task = values[0]
    tasks.append(task)
    # print('task = ', task)
    AppTime = values[1]
    apptimes.append(AppTime)
    # print('AppTime = ', AppTime)
    MPITime = values[2]
    mpitimes.append(MPITime)
    # print('MPITime = ', MPITime)
    MPITime_pc = values[3]
    precent.append(MPITime_pc)
    # print('MPITime_pc = ', MPITime_pc)
print('task:')
for t in tasks:
    print(t)
print('apptime:')
for t in apptimes:
    print(t)
print('mpitime:')
for t in mpitimes:
    print(t)
print('percent:')
for t in precent:
    print(t)