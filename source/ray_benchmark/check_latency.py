import argparse
import os
import numpy as np

def get_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--logdir", required=True)

    return parser.parse_args()

def main():
    args = get_args()
    
    files = os.listdir(args.logdir)

    warm_up = 30
    latencies = []
    for fname in files:
        c = 0 
        if fname.startswith("req_log"):
            with open(os.path.join(args.logdir, fname)) as in_file:
                for line in in_file:
                    if "Inference takes" in line:
                        if c > warm_up:
                            v = line.strip("\n").split("Inference takes")[1][:-2]
                            v = float(v)
                            latencies.append(v)
                        c += 1

    print(f"mean {np.mean(latencies)} ms, p50 {np.percentile(latencies, 50)}, p99 {np.percentile(latencies, 99)}")
    


if __name__ == "__main__":
    main()