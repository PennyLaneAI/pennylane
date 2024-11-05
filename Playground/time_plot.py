import sys
import timeit
import statistics
from functools import wraps
from memory_profiler import memory_usage
import argparse
import os.path

import pandas as pd
import matplotlib.pyplot as plt


import numpy as np
import pennylane as qml


parser = argparse.ArgumentParser(description='Quick benchmark')
# Working variables
parser.add_argument('--device', help='Lightning target device.', 
                    default='lightning.qubit',
                    type=str, required=False)
# Benchmark type argument
parser.add_argument('--benchmark',
                    choices=['time', 'memory', 'all', None],
                    default='all',
                    help='Benchmark type: time, memory, all, or None (default=all)')
# Times to repeat 
parser.add_argument('--times',
                    type=int, required=False,
                    default=3,
                    help='Times to repeat the time benchmark (default=3)')
# Output file argument
parser.add_argument('--output_file',
                    type=str, required=False,
                    default='',
                    help='Output file name (string)')
# Make plot argument
parser.add_argument('--make_plot',
                    action='store_true',
                    help='Flag to indicate whether to make a plot (default: False)')
# Data file to plot (reference) argument
parser.add_argument('--data_plot_ref',
                    type=str, default='',
                    help='Data file for reference plotting (string)')
# Data file to plot (current) argument
parser.add_argument('--data_plot_current',
                    type=str, default='',
                    help='Data file for current plotting (string)')

# Specific parameters for the benchmark 
parser.add_argument('--wires', nargs='+',
                    type=int, required=False,
                    help='list of wires')


# Set the profile global variables 
args = parser.parse_args()
# Target device
device_name = args.device
# Benchmark type [time, memory, all, None]
benchmark_type = args.benchmark
# Benchmark repetition 
benchmark_times = args.times
# Output file [str]
output_file = args.output_file
# Make plot [True, False]
make_plot = args.make_plot
# Data file 2 plot (reference) [str]
data_plot_ref = args.data_plot_ref
# Data file 2 plot (current) [str]
data_plot_current = args.data_plot_current

# Specific parameters
wires = args.wires

def test_qft(wires):
    """Test that the device can apply a multi-qubit QFT gate."""
    method = "mps"
    dev = qml.device("default.tensor", wires=wires, method=method, max_bond_dim=128)

    def circuit(basis_state):
        qml.BasisState(basis_state, wires=range(wires))
        qml.QFT(wires=range(wires))
        return qml.state()

    result = qml.QNode(circuit, dev)(np.array([0, 1] * (wires // 2)))
    
    return result


def code_to_benchmark(wires):
    result = test_qft(wires)
    print(result)

def timeit_decorator(runs=3):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            times = []
            for _ in range(runs):
                start_time = timeit.default_timer()
                func(*args, **kwargs)
                end_time = timeit.default_timer()
                times.append(end_time - start_time)
            
            max_time = max(times)
            min_time = min(times)
            avg_time = sum(times) / len(times)
            std_dev = statistics.stdev(times) if len(times) > 1 else 0.0
            
            return {
                'avg_time': avg_time,
                'std_dev': std_dev,
                'max_time': max_time,
                'min_time': min_time,
            }
        return wrapper
    return decorator

def memory_profiler_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Measure memory usage
        mem_usage = memory_usage((func, args, kwargs), max_usage=True)
        peak_memory = mem_usage
        
        if mem_usage is None:
            return {
                'peak_memory': 0.0
            }
            
                
        return {
            'peak_memory': peak_memory
        }
    return wrapper

@memory_profiler_decorator
def benchmark_memory_circuit(wires):
    if benchmark_type == 'all' or benchmark_type == 'memory':
        code_to_benchmark(wires)

@timeit_decorator(runs=benchmark_times)
def benchmark_time_circuit(wires):
    if benchmark_type == 'all' or benchmark_type == 'time':
        code_to_benchmark(wires)
    
def data_header():
    header = ''
    # Variable X
    header += 'wires '
    
    if benchmark_type == 'all':
        header += 'avg std_dev max min mem_peak '
    
    if benchmark_type == 'time':
        header += 'avg std_dev max min '

    if benchmark_type == 'memory':
        header += 'mem_peak '

    return header

def data_row(x, time, mem):
    
    row = str(x) + ' '
    
    if benchmark_type == 'all':
        row += ' '.join(str(t) for t in time)
        row += ' ' + str(mem)

    if benchmark_type == 'time':
        row += ' '.join(str(t) for t in time)

    if benchmark_type == 'memory':
        row += ' ' + str(mem)
        
    return row
    
def data2file(data):
    if output_file != '':
        with open(output_file, 'w') as bench_file: 
            bench_file.write(data)

def read_data(file_path):
    """Read the benchmarking data from a file."""
    if os.path.isfile(file_path):
        print("File read:",file_path)
    return pd.read_csv(file_path, sep='\s+')

def plot_comparison(data1, data2, label1, label2, output_file):
    """Plot comparison of average and memory peak."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting average values
    ax1.set_xlabel('wires')
    ax1.set_ylabel('Average time (s)', color='tab:blue')
    
    x = [int(i) for i in range(len(data1['wires']))]
    ax1.set_xticks(x)
    ax1.set_xticklabels(data1['wires'])

    # ax1.set_yscale('log')    
    ax1.plot(x, data1['avg'], marker='o', label=label1 + ' Avg time', color='tab:blue')
    ax1.plot(x, data2['avg'], marker='o', label=label2 + ' Avg time', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # Adding legends and title
    ax1.legend(loc='upper left')

    if benchmark_type == 'all' or benchmark_type == 'memory':
        # Creating a second y-axis for memory peaks
        ax2 = ax1.twinx()
        ax2.set_ylabel('Memory Peak (MiB)', color='tab:red')
        ax2.plot(x, data1['mem_peak'], marker='x', linestyle='--', label=label1 + ' Mem Peak', color='tab:blue')
        ax2.plot(x, data2['mem_peak'], marker='x', linestyle='--', label=label2 + ' Mem Peak', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')

        # Adding legends and title
        ax2.legend(loc='center left')

    # ax1.legend()
    # ax2.legend()
    plt.title('Timing comparison | ISAIC NV-A100')
    plt.grid()
    
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

def plot_speedup(data1, data2, output_file):
    """Plot comparison of average and memory peak."""
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plotting average values
    ax1.set_xlabel('wires')
    ax1.set_ylabel('Times faster', color='tab:blue')
    
    x = [int(i) for i in range(len(data1['wires']))]
    ax1.set_xticks(x)
    ax1.set_xticklabels(data1['wires'])

    speedup = data1['avg'] / data2['avg']
    # ax1.set_yscale('log')    
    ax1.plot(x, speedup, marker='o', label='Speedup', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    # Adding legends and title
    ax1.legend(loc='upper left')

    plt.title('Speed up comparison | ISAIC NV-A100')
    plt.grid()
    
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")


def main():
    if make_plot:
        # Read data
        if data_plot_current == '' or data_plot_ref == '':
            print("Data files are not set it correctly")
            return 2
        data1 = read_data(data_plot_ref)
        data2 = read_data(data_plot_current)

        # Plot comparison
        plot_comparison(data1, data2, 'OPENBLAS_NUM_THREADS = NA', 'OPENBLAS_NUM_THREADS = 1   ', 'plot_2.png')
        plot_speedup(data1, data2,'plot_2_speedup.png')
        
        return 0
        
    print(data_header())
    data_file = data_header() + '\n'
    for wire in wires:

        exec_time = benchmark_time_circuit(wire)
        exec_time = list(exec_time.values())

        mem_peak = benchmark_memory_circuit(wire)
    
        print(data_row(wire, exec_time, mem_peak['peak_memory']))
        data_file += data_row(wire, exec_time, mem_peak['peak_memory']) + '\n'
        
    data2file(data_file)
    
if __name__ == "__main__":
    main()