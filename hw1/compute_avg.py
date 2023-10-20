# Read data from the text file
with open('node2_proc24.txt', 'r') as file:
    lines = file.readlines()
n = 2
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
    elif 'Communication time' in line:
        communi.append(value)
        communi_num += value
    elif 'Elapsed time' in line:
        elapsed.append(value)
        elapsed_num += value

print('I/O = ', io_num / n)
print('Computation = ', comput_num / n)
print('Communication = ', communi_num / n)
print('Elapsed = ', elapsed_num / n)
