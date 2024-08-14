import numpy as np

def calculate_stats(filename):
    with open(filename, 'r') as file:
        times = [float(line.strip()) for line in file if line.strip()]
    
    times_array = np.array(times)
    mean = np.mean(times_array)
    std_dev = np.std(times_array)
    
    return mean, std_dev

files = ['policy_time', 'encoder_time']

for filename in files:
    mean, std_dev = calculate_stats(filename)
    print(f"{filename}:")
    print(f"  Mean: {mean:.6f} seconds")
    print(f"  Standard Deviation: {std_dev:.6f} seconds")
    print()