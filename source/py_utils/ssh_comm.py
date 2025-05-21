from pssh.clients.native.single import SSHClient
import time
from termcolor import colored
import threading
from typing import List

def get_host_ips(hostfile):
    host_ips = []
    host_slots = []
    with open(hostfile) as in_file:
        for line in in_file:
            if line.strip() != "":
                ip, slots = [v.strip() for v in line.split(' ')]
                host_ips += [ip]
                host_slots += [int(slots.split('=')[1])]
    return host_ips

def get_host_ips_slots(hostfile):
    host_ips = []
    host_slots = []
    with open(hostfile) as in_file:
        for line in in_file:
            if line.strip() != "":
                ip, slots = [v.strip() for v in line.split(' ')]
                host_ips += [ip]
                host_slots += [int(slots.split('=')[1])]
    return host_ips, host_slots

def get_storage_ips(storage_list_file):
    storage_ips = []
    with open(storage_list_file) as in_file:
        for i, line in enumerate(in_file):
            if i == 0 or i == 1 or line.strip() == "" or line.strip().startswith("//"):
                continue
            # Parse line: "IP, port"
            parts = line.split(',')
            if len(parts) >= 1:
                ip = parts[0].strip()
                storage_ips.append(ip)
    return storage_ips

def get_username(username_file="settings/username.txt"):
    try:
        with open(username_file) as f:
            for line in f:
                if line.strip() and not line.strip().startswith("//"):
                    return line.strip()
    except Exception as e:
        print(f"Error reading username file: {str(e)}")
        return None

def init_ssh_clients(host_ips):
    """
    Initialize SSH clients to connect to the given hosts.
    
    Args:
        host_ips (list): List of host IP addresses to connect to.
        
    Returns:
        list: List of initialized SSH clients.
    """
    # Get username from the settings file
    username = get_username()
    
    clients = []
    for ip in host_ips:
        cli = SSHClient(ip, user=username) if username else SSHClient(ip)  # Use username if available
        clients.append(cli)
        if username:
            print(f'connected to {ip} as user {username}')
        else:
            print(f'connected to {ip}')
    return clients

def __run_cmd(cli: SSHClient, cmd: str, timeout):
    output = cli.run_command(cmd, use_pty=True, read_timeout=timeout)
    print(f"{output.host} :: {cmd}")
    read_start_time = time.time()
    for line in output.stdout:
        print(f"{output.host} :: {line}")
    for line in output.stderr:
        print(colored(f'{output.host}::{line}', 'red'))
    # after read timeout
    if timeout is not None and time.time() - read_start_time > timeout:
        # close channel to exit
        cli.close_channel()

def parallel_exec_wait(clients, cmd: str, timeout=600): 
    """ default timeout 10min
    """
    cli_thds = []
    for c in clients:
        t = threading.Thread(target=__run_cmd, args=(c, cmd, timeout))
        t.start()
        cli_thds.append(t)

    for _, t in enumerate(cli_thds):
        t.join()

def parallel_exec_diff_cmd_wait(clients: List[SSHClient], cmds: List[str], timeout=600): 
    """ default timeout 10min
    """
    assert len(clients) == len(cmds)

    cli_thds = []
    for cli, cmd in zip(clients, cmds):
        t = threading.Thread(target=__run_cmd, args=(cli, cmd, timeout))
        t.start()
        cli_thds.append(t)

    for _, t in enumerate(cli_thds):
        t.join()