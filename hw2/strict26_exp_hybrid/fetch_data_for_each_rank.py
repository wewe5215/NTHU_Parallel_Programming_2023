# Read data from the text file
n = 1
p = 12
c = 1
current = 'n' + str(n) + 'p' + str(p) + 'c' + str(c)
filename = current + '.txt'
with open(filename, 'r') as file:
    lines = file.readlines()

io_num = 0
comput_num = 0
communi_num = 0
elapsed_num = 0
io = [ ]
comput = [ ]
communi = [ ]
elapsed = [ ]
# Process the data
for line in lines:
    line = line.split('\n')[0]
    parts = line.strip().split(' = ')
    if len(parts) == 2:
        key, value = parts[0], float(parts[1])
    if 'I/O time' in line:
        io.append(value)
        io_num += value
    elif 'Computation time' in line:
        comput.append(value)
        comput_num += value
    elif 'Commu' in line:
        communi.append(value)
        communi_num += value
    elif 'Elapsed time' in line:
        elapsed.append(value)
        elapsed_num += value
io_num = 1.8256
# communi_num = communi_num / p
# elapsed_num = elapsed_num / p
# print(io_num)
# print(elapsed_num - io_num - communi_num)
# print(communi_num)
# print(elapsed_num)
print("communication time")
for comm in communi:
    print(comm)
print("elapsed time")
for ela in elapsed:
    print(ela)
print("computation time")
for i in range(12):
    if(i != 0):
        print(elapsed[i] - communi[i])
    else:
        print(elapsed[i] - communi[i] - io_num)