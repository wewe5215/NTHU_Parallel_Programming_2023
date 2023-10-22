# Create a dictionary to store the total duration for each unique name

# Read the content from the file
for file_number in range(1, 13):
    filename = f'n1p12c1/out{file_number}.txt'
    with open(filename, 'r') as file:
        lines = file.readlines()
    name_duration_dict = {}
    # Define conversion factors for units
    unit_to_factor = {'s': 1, 'ms': 0.001, 'μs': 0.000001, 'ns' : 0.000000001}
    start_process_time = 0
    end_process_time = 0

    # Iterate through the lines and process the data
    for line in lines[1:]:  # Skip the header line
        parts = line.split('\t')  # Split the line by tab character

        name = parts[0]
        start = parts[1]
        duration = parts[2]  # Extract the duration

        # Remove the 's' unit symbol and convert to float
        start_value = float(start.replace('s', ''))
        
        # Extract the numerical part of the duration and unit
        duration_parts = duration.split()
        duration_value = float(duration_parts[0])
        duration_unit = duration_parts[1]

        if 'Init' in name:
            start_process_time = start_value
        elif 'Finalize' in name:
            end_process_time = start_value + duration_value * unit_to_factor.get(duration_unit, 1.0)

        # Convert the duration to seconds using the unit conversion factor
        duration_in_seconds = duration_value * unit_to_factor.get(duration_unit, 1.0)

        if 'open' in name or 'close' in name or 'read' in name or 'write' in name:
            if 'IO' in name_duration_dict:
                name_duration_dict['IO'] += duration_in_seconds
            else:
                name_duration_dict['IO'] = duration_in_seconds
        elif 'Sendrecv' in name:
            # print(duration_in_seconds)
            if 'Commu' in name_duration_dict:
                name_duration_dict['Commu'] += duration_in_seconds
            else:
                name_duration_dict['Commu'] = duration_in_seconds

    # Print the total durations for each unique name
    for name, total_duration in name_duration_dict.items():
        # print(f'{name} = {total_duration}')
        if 'IO' in name:
            print('I/O time = ', total_duration)
        elif 'Commu' in name:
            print('Communication time = ', total_duration)
    print('Elapsed time = ', end_process_time - start_process_time)