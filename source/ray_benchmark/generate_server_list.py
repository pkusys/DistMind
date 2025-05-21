import argparse
from ssh_comm import get_host_ips_slots

def get_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostfile", required=True)
    parser.add_argument("--output", default="server_list.txt")

    return parser.parse_args()

def main():
    """"""
    args = get_args()

    host_ips, slots = get_host_ips_slots(args.hostfile)
    
    with open(args.output, 'w') as out_file:
        for i in range(len(host_ips)):
            ip = host_ips[i]
            for gpu_idx in range(slots[i]):
                out_file.write(f"{ip}:{gpu_idx}\n")


if __name__ == "__main__":
    main()