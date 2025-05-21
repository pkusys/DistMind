import os
import re
import glob
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Define the base directory
base_dir = './tmp/test1'
systems = ['gpu', 'distmind_cache', 'distmind_remote', 'mps', 'ray']
model_names = ['bert', 'den', 'gpt', 'inc', 'res']

# Model name mapping to full names
model_full_names = {
    'bert': 'BERT',
    'den': 'DenseNet',
    'gpt': 'GPT-2',
    'inc': 'InceptionV3',
    'res': 'ResNet152'
}

# Data structure to store the results
latencies = defaultdict(dict)  # {model_name: {system_name: latency}}

# Process each system
for system in systems:
    system_dir = os.path.join(base_dir, system)
    if not os.path.exists(system_dir):
        print(f"Warning: {system_dir} does not exist. Skipping...")
        continue
    
    for model in model_names:
        model_dir = os.path.join(system_dir, model)
        if not os.path.exists(model_dir):
            print(f"Warning: {model_dir} does not exist. Skipping...")
            continue
        
        if system in ['gpu', 'distmind_cache', 'distmind_remote']:
            # Extract Total Latency from log_worker files
            total_latencies = []
            worker_logs = glob.glob(os.path.join(model_dir, "log_worker_*.txt"))
            
            for log_file in worker_logs:
                if not os.path.exists(log_file):
                    continue
                
                latency_values = []
                with open(log_file, 'r') as f:
                    for line in f:
                        match = re.search(r'Total Latency: ([\d.]+) ms', line)
                        if match:
                            latency_values.append(float(match.group(1)))
                
                if latency_values:
                    # Add the average latency from this worker to our list
                    total_latencies.append(np.mean(latency_values))
            
            if total_latencies:
                # Average across all workers
                latencies[model][system] = np.mean(total_latencies)
            
        elif system == 'mps':
            # Extract inference time from MPS logs
            inference_logs_dir = os.path.join(model_dir, "inference_logs")
            if not os.path.exists(inference_logs_dir):
                continue
                
            inference_times = []
            log_files = glob.glob(os.path.join(inference_logs_dir, "*.inf.log"))
            
            for log_file in log_files:
                with open(log_file, 'r') as f:
                    for line in f:
                        match = re.search(r'inference time: ([\d.]+)', line)
                        if match:
                            inference_times.append(float(match.group(1)) * 1000)  # Convert to ms
            
            if inference_times:
                latencies[model][system] = np.mean(inference_times)
                
        elif system == 'ray':
            # Extract inference time from Ray server logs
            log_server = os.path.join(model_dir, "log_server.txt")
            if not os.path.exists(log_server):
                continue
                
            inference_times = []
            with open(log_server, 'r') as f:
                for line in f:
                    match = re.search(r'inference done, cost ([\d.]+) ms', line)
                    if match:
                        inference_times.append(float(match.group(1)))
            
            if inference_times:
                latencies[model][system] = np.mean(inference_times)

# Print the collected data for verification
print("Collected Latency Data (ms):")
for model in model_names:
    if model in latencies:
        print(f"{model}: {latencies[model]}")

# Create the bar chart
# Define a consistent order for systems in the plot
plot_systems = [sys for sys in systems if any(sys in latencies[model] for model in latencies)]

# Set up the plot
fig, ax = plt.subplots(figsize=(12, 8))
x = np.arange(len(latencies))
width = 0.15  # Width of bars
multiplier = 0

# Define colors for systems
colors = {
    'gpu': 'tab:blue',
    'distmind_cache': 'tab:orange',
    'distmind_remote': 'tab:green',
    'mps': 'tab:red',
    'ray': 'tab:purple'
}

# Plot each system
for system in plot_systems:
    latency_values = []
    for model in latencies:
        if system in latencies[model]:
            latency_values.append(latencies[model][system])
        else:
            latency_values.append(0)
    
    offset = width * multiplier
    rects = ax.bar(x + offset, latency_values, width, label=system, color=colors.get(system, None))
    multiplier += 1

# Set y-axis to logarithmic scale
ax.set_yscale('log', base=10)

# Set fixed y-ticks at powers of 10
import matplotlib.ticker as ticker
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:g}'.format(y)))
ax.yaxis.set_major_locator(ticker.FixedLocator([10, 100, 1000]))
ax.set_ylim(bottom=10, top=1000)  # Set fixed limits to ensure ticks are visible

# Add labels and legend
ax.set_xlabel('Model')
ax.set_ylabel('Latency (ms)')
ax.set_title('Inference latency with different solutions')
ax.set_xticks(x + width * (len(plot_systems) - 1) / 2)

# Use full model names for x-axis labels
model_labels = [model_full_names.get(model, model) for model in latencies.keys()]
ax.set_xticklabels(model_labels)
ax.legend(loc='best')

# Add a grid for easier reading
ax.grid(True, axis='y', linestyle='--', alpha=0.7)

# Save the figure
plt.tight_layout()
plt.savefig('fig6.png')
plt.close()

print("Plot has been saved")