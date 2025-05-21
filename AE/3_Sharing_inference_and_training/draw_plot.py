import os
import numpy as np
import matplotlib.pyplot as plt

def read_bounds(bounds_file):
    """Read throughput bounds from the bounds.txt file"""
    bounds = {}
    try:
        with open(bounds_file, 'r') as f:
            # Skip header line
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    system = parts[0].strip()
                    bound_type = parts[1].strip()
                    value = float(parts[2].strip())
                    key = f"{system}_{bound_type}"
                    bounds[key] = value
        return bounds
    except Exception as e:
        print(f"Error reading bounds file: {e}")
        return {}

def read_throughput_data(throughput_file):
    """Read throughput data from a throughput file"""
    client_throughputs = []
    train_throughputs = []
    
    try:
        with open(throughput_file, 'r') as f:
            # Skip header line
            next(f)
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 2:
                    client_tp = float(parts[0].strip())
                    train_tp = float(parts[1].strip())
                    client_throughputs.append(client_tp)
                    train_throughputs.append(train_tp)
        return client_throughputs, train_throughputs
    except Exception as e:
        print(f"Error reading throughput file {throughput_file}: {e}")
        return [], []

def plot_utilization(system_name, client_utilization, train_utilization, output_file):
    """Create an area plot of inference and training utilization"""
    plt.figure(figsize=(10, 6))
    
    # Create time indices (x-axis)
    x = range(len(client_utilization))
    
    # Plot inference utilization (blue)
    plt.fill_between(x, client_utilization, alpha=0.7, color='blue', label='Inference Utilization')
    
    # Check if train utilization has non-zero values
    has_training = any(tp > 0 for tp in train_utilization)
    
    # Plot train utilization stacked on top of inference (red) only if there is training
    if has_training:
        plt.fill_between(x, client_utilization, [client_utilization[i] + train_utilization[i] for i in x], 
                        alpha=0.7, color='red', label='Training Utilization')
    
    plt.title(f'{system_name} Utilization Over Time')
    plt.xlabel('Time Index')
    plt.ylabel('Utilization')
    
    # Set y-axis limit based on whether training exists
    if has_training:
        plt.ylim(0, max(1.1, max([client_utilization[i] + train_utilization[i] for i in x])))
    else:
        plt.ylim(0, max(1.1, max(client_utilization)))
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}" + ('' if has_training else ' (no training data)'))
    plt.close()

def main():
    # Base directory for data files
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'tmp/test3')
    
    # Path to bounds file
    bounds_file = os.path.join(base_dir, 'bounds.txt')
    
    # Read bounds
    bounds = read_bounds(bounds_file)
    if not bounds:
        print("Failed to read bounds data. Exiting.")
        return
    
    print("Throughput Bounds:")
    for key, value in bounds.items():
        print(f"  {key}: {value}")
    
    # Get required bounds
    gpu_inference_bound = bounds.get('GPU_Inference', 0)
    distmind_train_bound = bounds.get('DistMind_Train', 0)
    mps_train_bound = bounds.get('MPS_Train', 0)
    ray_train_bound = bounds.get('Ray_Train', 0)
    
    if gpu_inference_bound == 0:
        print("ERROR: GPU inference bound is zero or not found")
        return
    
    # Paths to throughput files
    throughput_files = {
        'DistMind': os.path.join(base_dir, 'distmind_throughput.txt'),
        'GPU': os.path.join(base_dir, 'gpu_throughput.txt'),
        'MPS': os.path.join(base_dir, 'mps_throughput.txt'),
        'Ray': os.path.join(base_dir, 'ray_throughput.txt')
    }
    
    # Process each system
    for system, tp_file in throughput_files.items():
        # Read throughput data
        client_tp, train_tp = read_throughput_data(tp_file)
        
        if not client_tp:
            print(f"No throughput data for {system}. Skipping.")
            continue
        
        print(f"\nProcessing {system} data:")
        print(f"  Client Throughput Points: {len(client_tp)}")
        print(f"  Train Throughput Points: {len(train_tp)}")
        
        # Calculate utilization
        client_utilization = [tp / gpu_inference_bound for tp in client_tp]
        
        # Check if all training throughput values are zero
        all_train_zero = all(tp == 0 for tp in train_tp)
        
        # Select appropriate train bound based on system
        if system == 'DistMind':
            train_bound = distmind_train_bound
        elif system == 'MPS':
            train_bound = mps_train_bound
        elif system == 'Ray':
            train_bound = ray_train_bound
        else:
            train_bound = 1.0  # Default or for GPU (which should have 0 train throughput)
        
        if train_bound == 0:
            print(f"WARNING: {system} train bound is zero, using 1.0 instead")
            train_bound = 1.0
        
        # Calculate train utilization if there's any training data
        if all_train_zero:
            train_utilization = [0] * len(client_utilization)
            print(f"Note: {system} has no training data (all values are zero)")
        else:
            train_utilization = [tp / train_bound for tp in train_tp]
        
        # Create and save plot
        output_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'{system.lower()}_utilization.png')
        plot_utilization(system, client_utilization, train_utilization, output_file)

if __name__ == "__main__":
    main()