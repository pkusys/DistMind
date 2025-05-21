import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

import sys
from py_utils.check_client import extract_last_block_metrics

# Define the base directory
base_dir = './tmp/test2'
systems = ['gpu', 'distmind', 'mps', 'ray']

# Data structures to store the parsed results
zipf_throughput = defaultdict(lambda: defaultdict(list))
resched_throughput = defaultdict(lambda: defaultdict(list))
zipf_latency = defaultdict(lambda: defaultdict(list))
resched_latency = defaultdict(lambda: defaultdict(list))
zipf_latency_50th = defaultdict(lambda: defaultdict(list))
zipf_latency_99th = defaultdict(lambda: defaultdict(list))
resched_latency_50th = defaultdict(lambda: defaultdict(list))
resched_latency_99th = defaultdict(lambda: defaultdict(list))

# GPU bounds
gpu_avg_throughput = 0
gpu_avg_latency = 0
gpu_count = 0

# Parse log files and extract metrics
for system in systems:
    system_dir = os.path.join(base_dir, system)
    
    if system != 'ray':
        log_files = glob.glob(os.path.join(system_dir, 'log_client_*.txt'))
        
        for log_file in log_files:
            # Extract zipf and resched values from filename
            filename = os.path.basename(log_file)
            zipf_match = re.search(r'zipf([\d.]+)', filename)
            resched_match = re.search(r'resched([\d.]+)', filename)
            
            # Clean up any trailing periods that might be present in the extracted values
            zipf_str = zipf_match.group(1) if zipf_match else None
            resched_str = resched_match.group(1) if resched_match else None
            
            if zipf_str and zipf_str.endswith('.'):
                zipf_str = zipf_str[:-1]
            if resched_str and resched_str.endswith('.'):
                resched_str = resched_str[:-1]
            
            if zipf_match and resched_match:
                zipf_value = float(zipf_str)
                resched_value = float(resched_str)
                
                # Extract metrics from the log file
                metrics = extract_last_block_metrics(log_file)
                
                if metrics:
                    throughput = metrics["Average Throughput"]
                    latency = metrics["Average Latency"]
                    latency_50th = metrics["50th Latency"]
                    latency_99th = metrics["99th Latency"]
                    
                    if throughput is not None and latency is not None:
                        # Store data by zipf value
                        zipf_throughput[zipf_value][system].append(throughput)
                        zipf_latency[zipf_value][system].append(latency)
                        if latency_50th is not None:
                            zipf_latency_50th[zipf_value][system].append(latency_50th)
                        if latency_99th is not None:
                            zipf_latency_99th[zipf_value][system].append(latency_99th)
                        
                        # Store data by resched value
                        resched_throughput[resched_value][system].append(throughput)
                        resched_latency[resched_value][system].append(latency)
                        if latency_50th is not None:
                            resched_latency_50th[resched_value][system].append(latency_50th)
                        if latency_99th is not None:
                            resched_latency_99th[resched_value][system].append(latency_99th)
                        
                        # Calculate GPU bounds
                        if system == 'gpu':
                            gpu_avg_throughput += throughput
                            gpu_avg_latency += latency
                            gpu_count += 1
    else:
        # Process Ray data differently - from subdirectories
        ray_dirs = [d for d in os.listdir(system_dir) if os.path.isdir(os.path.join(system_dir, d))]
        
        for ray_dir in ray_dirs:
            # Extract zipf and resched values from directory name
            zipf_match = re.search(r'zipf([\d.]+)', ray_dir)
            resched_match = re.search(r'resched([\d.]+)', ray_dir)
            
            if zipf_match and resched_match:
                zipf_value = float(zipf_match.group(1))
                resched_value = float(resched_match.group(1))
                
                # Read latency data from check_latency.txt
                latency_file = os.path.join(system_dir, ray_dir, 'check_latency.txt')
                if os.path.exists(latency_file):
                    with open(latency_file, 'r') as f:
                        latency_content = f.read()
                        # Extract mean, p50, p99 from the latency file
                        mean_match = re.search(r'mean ([\d.]+)', latency_content)
                        p50_match = re.search(r'p50 ([\d.]+)', latency_content)
                        p99_match = re.search(r'p99 ([\d.]+)', latency_content)
                        
                        if mean_match and p50_match and p99_match:
                            latency = float(mean_match.group(1))
                            p50 = float(p50_match.group(1))
                            p99 = float(p99_match.group(1))
                            
                            # Read throughput data from avg_stats.txt
                            throughput_file = os.path.join(system_dir, ray_dir, 'avg_stats.txt')
                            if os.path.exists(throughput_file):
                                with open(throughput_file, 'r') as f:
                                    throughput_lines = f.readlines()
                                    
                                    # Get the last throughput value only
                                    throughput = 0
                                    if throughput_lines:
                                        last_line = throughput_lines[-1]
                                        if last_line.startswith('total avg'):
                                            parts = last_line.split(',')[0].split()
                                            if len(parts) >= 3:
                                                try:
                                                    throughput = float(parts[2])
                                                except ValueError:
                                                    pass
                                    
                                    # Store data by zipf value
                                    zipf_throughput[zipf_value][system].append(throughput)
                                    zipf_latency[zipf_value][system].append(latency)
                                    zipf_latency_50th[zipf_value][system].append(p50)
                                    zipf_latency_99th[zipf_value][system].append(p99)
                                    
                                    # Store data by resched value
                                    resched_throughput[resched_value][system].append(throughput)
                                    resched_latency[resched_value][system].append(latency)
                                    resched_latency_50th[resched_value][system].append(p50)
                                    resched_latency_99th[resched_value][system].append(p99)

# Calculate GPU bounds as averages
if gpu_count > 0:
    gpu_avg_throughput /= gpu_count
    gpu_avg_latency /= gpu_count

# Calculate averages for each category
for zipf in zipf_throughput:
    for system in zipf_throughput[zipf]:
        zipf_throughput[zipf][system] = np.mean(zipf_throughput[zipf][system])
        
for zipf in zipf_latency:
    for system in zipf_latency[zipf]:
        zipf_latency[zipf][system] = np.mean(zipf_latency[zipf][system])
        
for zipf in zipf_latency_50th:
    for system in zipf_latency_50th[zipf]:
        zipf_latency_50th[zipf][system] = np.mean(zipf_latency_50th[zipf][system])
        
for zipf in zipf_latency_99th:
    for system in zipf_latency_99th[zipf]:
        zipf_latency_99th[zipf][system] = np.mean(zipf_latency_99th[zipf][system])
        
for resched in resched_throughput:
    for system in resched_throughput[resched]:
        resched_throughput[resched][system] = np.mean(resched_throughput[resched][system])
        
for resched in resched_latency:
    for system in resched_latency[resched]:
        resched_latency[resched][system] = np.mean(resched_latency[resched][system])
        
for resched in resched_latency_50th:
    for system in resched_latency_50th[resched]:
        resched_latency_50th[resched][system] = np.mean(resched_latency_50th[resched][system])
        
for resched in resched_latency_99th:
    for system in resched_latency_99th[resched]:
        resched_latency_99th[resched][system] = np.mean(resched_latency_99th[resched][system])

# Sort the keys for consistent plotting
zipf_keys = sorted(zipf_throughput.keys())
resched_keys = sorted(resched_throughput.keys())

# Define a function to plot grouped bar charts
def plot_grouped_bars(x_values, data, systems_to_plot, bound_value, title, xlabel, ylabel, filename, percentile_50th=None, percentile_99th=None):
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bar_width = 0.25  # Reduced from 0.35 to accommodate three bars
    index = np.arange(len(x_values))
    
    colors = {'distmind': 'blue', 'mps': 'orange', 'ray': 'green'}
    
    # Plot bars for each system
    for i, system in enumerate(systems_to_plot):
        values = [data[x][system] if system in data[x] else 0 for x in x_values]
        bars = ax.bar(index + i * bar_width, values, bar_width, label=system.upper(), color=colors[system])
        
        # Add error bars for 50th and 99th percentiles if provided (for latency plots)
        if percentile_50th and percentile_99th:
            for j, x in enumerate(x_values):
                if system in percentile_50th[x] and system in percentile_99th[x]:
                    # Calculate the error bar heights relative to the average
                    p50 = percentile_50th[x][system]
                    p99 = percentile_99th[x][system]
                    avg = values[j]
                    
                    # Plot vertical lines for 50th and 99th percentiles
                    # Line from 50th to 99th percentile
                    ax.plot([index[j] + i * bar_width, index[j] + i * bar_width], 
                            [p50, p99], 
                            'k-', lw=1.5)
                    
                    # Small horizontal line at 50th percentile
                    ax.plot([index[j] + i * bar_width - 0.05, index[j] + i * bar_width + 0.05], 
                            [p50, p50], 
                            'k-', lw=1.5)
                    
                    # Small horizontal line at 99th percentile
                    ax.plot([index[j] + i * bar_width - 0.05, index[j] + i * bar_width + 0.05], 
                            [p99, p99], 
                            'k-', lw=1.5)
    
    # Plot the bound line
    ax.axhline(y=bound_value, color='r', linestyle='-', label='GPU Bound')
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(index + bar_width / 2)
    
    # Format X-axis labels
    x_labels = [str(x) for x in x_values]
    ax.set_xticklabels(x_labels)
    
    # Add a legend with extra entries for 50th and 99th percentiles if needed
    handles, labels = ax.get_legend_handles_labels()
    if percentile_50th and percentile_99th:
        import matplotlib.lines as mlines
        percentile_line = mlines.Line2D([], [], color='black', marker='_', 
                                        linestyle='-', markersize=10, label='50th-99th Percentile')
        handles.append(percentile_line)
    
    ax.legend(handles=handles)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot the results
systems_to_plot = ['distmind', 'mps', 'ray']  # Plot these three systems

output_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs(output_dir, exist_ok=True)

# Plot 1: Throughput vs Zipf
plot_grouped_bars(
    zipf_keys,
    zipf_throughput,
    systems_to_plot,
    gpu_avg_throughput,
    'fig7_a Performance under different distributions',
    'Zipf Value',
    'Throughput (rps)',
    os.path.join(output_dir, 'fig7_a.png')
)

# Plot 2: Throughput vs Resched
plot_grouped_bars(
    resched_keys,
    resched_throughput,
    systems_to_plot,
    gpu_avg_throughput,
    'fig8_a Performance under different scheduling cycles',
    'Resched Value',
    'Throughput (rps)',
    os.path.join(output_dir, 'fig8_a.png')
)

# Plot 3: Latency vs Zipf
plot_grouped_bars(
    zipf_keys,
    zipf_latency,
    systems_to_plot,
    gpu_avg_latency,
    'fig7_b Performance under different distributions',
    'Zipf Value',
    'Latency (ms)',
    os.path.join(output_dir, 'fig7_b.png'),
    zipf_latency_50th,
    zipf_latency_99th
)

# Plot 4: Latency vs Resched
plot_grouped_bars(
    resched_keys,
    resched_latency,
    systems_to_plot,
    gpu_avg_latency,
    'fig8_b Performance under different scheduling cycles',
    'Resched Value',
    'Latency (ms)',
    os.path.join(output_dir, 'fig8_b.png'),
    resched_latency_50th,
    resched_latency_99th
)

print("Plots have been generated successfully!")