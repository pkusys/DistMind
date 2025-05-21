import argparse
from ssh_comm import get_host_ips, get_storage_ips, get_host_ips_slots, init_ssh_clients, parallel_exec_diff_cmd_wait, parallel_exec_wait, get_username

public_cmd = " ; ".join([
            "if [ -z \"$SHELL\" ]; then bash -l; fi",
            "source ~/.bashrc",
            "cd ~/DistMindAE/distmind",
            "source settings/config.sh"
        ])

def get_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostfile", default="settings/serverhost_list.txt", help="hostfile")
    parser.add_argument("--storage_list_file", default="settings/storage_list.txt", help="storage list file")
    parser.add_argument("--launch_part", required=True, help="launch part")
    parser.add_argument("--log_file", help="log file")
    parser.add_argument("--log_dir", help="log dir")
    parser.add_argument("--stop", action="store_true", help="stop the cluster")
    parser.add_argument("--test_index", type=int, help="test index")
    parser.add_argument("--first", action="store_true", help="only use first host when getting output")

    parser.add_argument("--n_models", type=int, help="number of models")
    parser.add_argument("--model_seed", help="model seed")
    parser.add_argument("--system_type", help="system type")
    parser.add_argument("--cache_size", type=int, default=4096000000, help="cache size")
    parser.add_argument("--cache_block_size", type=int, default=4096000, help="cache block size")
    parser.add_argument("--training-log-dir", help="training log dir")
    parser.add_argument("--inference-log-dir", help="inference log dir")
    return parser.parse_args()

def launch_storage_client(hostfile, is_stop=False):
    host_ips, slots = get_host_ips_slots(hostfile)
    ssh_clients = init_ssh_clients(host_ips)
    cmds = []
    if is_stop:
        for i, slot in enumerate(slots):
            # Command to stop the server
            cmd = " ; ".join([
                public_cmd,
                "if [ -f .pid_storage_client ]; then kill $(cat .pid_storage_client) && rm .pid_storage_client && echo 'Storage client process killed'; else echo 'No storage client process found'; fi"
            ])
            cmds.append(cmd)
    else:
        for i, slot in enumerate(slots):
            # Main command with shell check prepended
            cmd = " ; ".join([
                public_cmd,
                f"./scripts/launch_store_client.sh {i}",
            ])
            cmds.append(cmd)

    # Use different commands for each client
    parallel_exec_diff_cmd_wait(ssh_clients, cmds, timeout=None)

    if is_stop:
        print("Storage client stopped.")
    else:
        print("Storage client started.")

def launch_ray_server(test_index, hostfile, n_models, model_seed, logfile, is_stop=False):
    host_ips, slots = get_host_ips_slots(hostfile)
    ssh_clients = init_ssh_clients(host_ips)
    cmds = []
    if is_stop:
        for i, slot in enumerate(slots):
            # Command to stop the server
            cmd = " ; ".join([
                public_cmd,
                "ps aux | grep ray | grep -v grep | awk '{print $2}' | xargs -r kill -9",
                "if [ -f .pid_ray_server ]; then kill $(cat .pid_ray_server) && rm .pid_ray_server && echo 'Ray server process killed'; else echo 'No Ray server process found'; fi",
            ])
            cmds.append(cmd)
    else:
        for i, slot in enumerate(slots):
            # Main command with shell check prepended
            if logfile is None:
                logfile = f"tmp/test{test_index}/ray/{model_seed}/log_server.txt"
            cmd = " ; ".join([
                public_cmd,
                "ps aux | grep ray | grep -v grep | awk '{print $2}' | xargs -r kill -9",
                f"python ./source/ray_benchmark/pt_example.py --n_gpus {slot} --n_model_variants  {n_models} --model_name {model_seed}  > {logfile} 2>&1 &",
                "echo $! > .pid_ray_server",
            ])
            cmds.append(cmd)

    # Use different commands for each client
    parallel_exec_diff_cmd_wait(ssh_clients, cmds, timeout=None)

    if is_stop:
        print("Ray server stopped.")
    else:
        print("Ray server started.")

def launch_mps_storage(hostfile, n_models, model_seed, is_stop=False):
    host_ips, slots = get_host_ips_slots(hostfile)
    ssh_clients = init_ssh_clients(host_ips)

    if is_stop:
        cmd = " ; ".join([
            public_cmd,
            "./scripts/mps/run_shutdown.sh"
        ])
    else:
        cmd = " ; ".join([
            public_cmd,
            "./scripts/mps/run_mps.sh"
            "sleep 10 # Give MPS a bit of time to start"
            f"./scripts/mps/run_generate.sh {n_models} {model_seed}",
            "./scripts/mps/run_load_models.sh"
        ])

    parallel_exec_wait(ssh_clients, cmd, timeout=None)

    if is_stop:
        print("MPS storage stopped.")
    else:
        print("MPS storage started.")

def launch_mps_server(test_index, hostfile, n_models, model_seed, system_type,
    training_log_dir, inference_log_dir, is_stop=False):
    host_ips, slots = get_host_ips_slots(hostfile)
    ssh_clients = init_ssh_clients(host_ips)

    if training_log_dir is None:
        training_log_dir = "None"
    if inference_log_dir is None:
        inference_log_dir = "None"
    if is_stop:
        cmd = " ; ".join([
            public_cmd,
            "./scripts/mps/shutdown_mps_server.sh"
        ])
    else:
        cmd = " ; ".join([
            public_cmd,
            f"./scripts/mps/launch_mps_server_part.sh {test_index} {n_models} {model_seed} {training_log_dir} {inference_log_dir}",
        ])

    parallel_exec_wait(ssh_clients, cmd, timeout=None)

    if is_stop:
        print("MPS server stopped.")
    else:
        print("MPS server started.")

def launch_dist_server(test_index, hostfile, n_models, model_seed, system_type, 
    cache_size, cache_block_size, is_stop=False):
    host_ips, slots = get_host_ips_slots(hostfile)
    ssh_clients = init_ssh_clients(host_ips)

    if is_stop:
        cmd = " ; ".join([
            public_cmd,
            "./scripts/shutdown_distmind_server_part.sh"
        ])
    else:
        cmd = " ; ".join([
            public_cmd,
            f"./scripts/launch_distmind_server_part.sh {test_index} {n_models} {model_seed} {system_type} {cache_size} {cache_block_size}",
        ])

    parallel_exec_wait(ssh_clients, cmd, timeout=None)

    if is_stop:
        print("Distributed server stopped.")
    else:
        print("Distributed server started.")

def launch_dist_storage(storage_list_file, log_file, is_stop=False):
    """
    Launch or stop distributed storage servers on each storage IP.
    
    Args:
        storage_list_file (str): Path to the storage list file
        is_stop (bool): Whether to stop the storage servers (True) or start them (False)
    """
    
    # Get the storage IPs from the storage list file
    storage_ips = get_storage_ips(storage_list_file)
    if not storage_ips:
        print("No storage IPs found in the storage list file.")
        return
    
    # Initialize SSH clients for each storage IP
    ssh_clients = init_ssh_clients(storage_ips)
    
    # Prepare commands for each storage IP
    cmds = []
    for ip in storage_ips:
        if is_stop:
            # Command to stop the storage server
            cmd = " ; ".join([
                public_cmd,
                "if [ -f .pid_storage ]; then kill $(cat .pid_storage) && rm .pid_storage && echo 'Storage process killed'; else echo 'No storage process found'; fi"
            ])
        else:
            # Command to start the storage server
            cmd = " ; ".join([                
                public_cmd,
                "STORAGE_PORT=$GLOBAL_STORAGE_PORT",
                "mkdir -p tmp/test1/$SYSTEM_TYPE/$MODEL_SEED",
                f"./build/bin/storage $STORAGE_PORT $STORAGE_PORT > {log_file} 2>&1 &",
                "echo $! > .pid_storage",
                "echo \"Started storage server with PID $(cat .pid_storage)\""
            ])
        cmds.append(cmd)
    
    # Execute commands on each storage IP
    parallel_exec_diff_cmd_wait(ssh_clients, cmds, timeout=None)
    
    if is_stop:
        print("Storage servers stopped.")
    else:
        print("Storage servers started.")

def create_log_directories(hostfile, storage_list_file, log_dir, is_stop=False):
    """
    Create log directories on all hosts specified in hostfile and storage_list_file
    
    Args:
        hostfile (str): Path to the host file with compute nodes
        storage_list_file (str): Path to the storage list file
        log_dir (str): Path to the log directory to create
        is_stop (bool): Whether to remove the log directory (True) or create it (False)
    """
    from ssh_comm import get_host_ips, get_storage_ips, init_ssh_clients, parallel_exec_wait
    
    # Get all unique IPs from both compute and storage hosts
    host_ips = get_host_ips(hostfile)
    storage_ips = get_storage_ips(storage_list_file)
    
    # Combine and deduplicate IPs
    all_ips = list(set(host_ips + storage_ips))
    print(f"Creating log directory on {len(all_ips)} hosts")
    
    # Initialize SSH clients for all IPs
    ssh_clients = init_ssh_clients(all_ips)
    
    # Command to create log directory
    if is_stop:
        cmd = " ; ".join([
            public_cmd,
            f"rm -rf {log_dir}",
            f"echo 'Removed directory {log_dir}'"
        ])
    else:
        # Command to create the log directory
        cmd = " ; ".join([
            public_cmd,
            f"mkdir -p {log_dir}",
            f"echo 'Created directory {log_dir}'"
        ])
    
    # Execute command on all hosts
    parallel_exec_wait(ssh_clients, cmd, timeout=None)
    
    print(f"Log directories created at {log_dir} on all hosts")

def launch_cleanup(hostfile, storage_list_file):
    host_ips, slots = get_host_ips_slots(hostfile)
    storage_ips = get_storage_ips(storage_list_file)
    all_ips = list(set(host_ips + storage_ips))
    print(f"Cleaning up on {len(all_ips)} hosts")
    ssh_clients = init_ssh_clients(all_ips)
    cmd = " ; ".join([
        public_cmd,
        "find /dev/shm/* -writable -exec rm -rf {} + 2>/dev/null  || true",
    ])
    parallel_exec_wait(ssh_clients, cmd, timeout=None)
    print("Cleanup completed.")

def get_remote_output(hostfile, log_dir, first=False):
    """
    Copy log files from remote hosts to the local machine.
    
    Args:
        hostfile (str): Path to the host file containing the remote hosts
        log_dir (str): Path to the log directory on remote and local hosts
        first (bool): If True, only copy from the first host in the hostfile
    """
    import os
    import subprocess
    
    username = get_username()
    # Create local log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Get the host IPs from the hostfile
    host_ips = get_host_ips(hostfile)
    if not host_ips:
        print("No host IPs found in the hostfile.")
        return
    
    # If first is True, only use the first host
    if first:
        host_ips = [host_ips[0]]
    
    print(f"Copying log files from {len(host_ips)} hosts to {log_dir}")
    
    # Copy log files from each host
    for ip in host_ips:
        try:
            # Create a directory for each host's logs
            #host_dir = os.path.join(log_dir, ip.replace(".", "_"))
            #os.makedirs(host_dir, exist_ok=True)
            host_dir = log_dir
            
            # Use scp to copy files from remote host to local machine
            if username:
                cmd = ["scp", "-r", f"{username}@{ip}:{log_dir}/*", host_dir]
            else:
                cmd = ["scp", "-r", f"{ip}:{log_dir}/*", host_dir]
            print(f"Executing: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"Successfully copied logs from {ip} to {host_dir}")
            else:
                print(f"Failed to copy logs from {ip}: {result.stderr}")
        except Exception as e:
            print(f"Error copying logs from {ip}: {str(e)}")
    
    print("Log file copy completed.")

def main():
    """"""
    args = get_args()
    
    if args.launch_part == "ray_server":
        launch_ray_server(args.test_index, args.hostfile, args.n_models, args.model_seed, args.log_file, args.stop)
    elif args.launch_part == "dist_storage":
        launch_dist_storage(args.storage_list_file, args.log_file, args.stop)
    elif args.launch_part == "create_dir":
        create_log_directories(args.hostfile, args.storage_list_file, args.log_dir)
    elif args.launch_part == "mps_storage":
        launch_mps_storage(args.hostfile, args.n_models, args.model_seed, args.stop)
    elif args.launch_part == "mps_server":
        launch_mps_server(args.test_index, args.hostfile, args.n_models, args.model_seed, args.system_type, args.training_log_dir, args.inference_log_dir, args.stop)
    elif args.launch_part == "dist_server":
        launch_dist_server(args.test_index, args.hostfile, args.n_models, args.model_seed, args.system_type, args.cache_size, args.cache_block_size, args.stop)
    elif args.launch_part == "storage_client":
        launch_storage_client(args.hostfile, args.stop)
    elif args.launch_part == "cleanup":
        launch_cleanup(args.hostfile, args.storage_list_file)
    elif args.launch_part == "get_output":
        # Determine whether to get logs from all hosts or just the first one
        first_only = args.first
        get_remote_output(args.hostfile, args.log_dir, first_only)
    else:
        print(f"Unknown launch part: {args.launch_part}")
        return

if __name__ == "__main__":
    main()