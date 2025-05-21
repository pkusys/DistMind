import argparse

def get_args():
    """"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True)
    parser.add_argument("--output", required=True)

    return parser.parse_args()

def main():
    args = get_args()
    agg_stats = []
    assert args.log != args.output, "output file and input log have to be different"

    for_avg_tputs = []
    with open(args.log, 'r') as in_file:
        for line in in_file:
            entry = [float(e) for e in line.strip().strip('\n').split(",")]
            timestamp = entry[0]
            inf_throughput = 0
            train_throughput = 0
            for i in range(1, len(entry)):
                v = entry[i]
                if i % 2 == 1:
                    inf_throughput += v
                else:
                    train_throughput += v

            agg_stats.append(", ".join([str(v) for v in [
                timestamp, inf_throughput, train_throughput
            ]]))
            for_avg_tputs.append([
                timestamp, inf_throughput, train_throughput
            ])
    
    with open(args.output, 'w') as out_file:
        for e in agg_stats:
            out_file.write(f"{e}\n")

    warm_up = 10
    total_inf = 0
    total_train = 0
    total_td = 0
    for i, e in enumerate(for_avg_tputs):
        if i > 0:
            td = e[0] - for_avg_tputs[i-1][0]
            inf_avg = e[1] / td
            train_avg = e[2] / td
            print(f"avg tput {inf_avg}, {train_avg}")
            if i > warm_up and i < len(for_avg_tputs) - 1:
                total_inf += e[1]
                total_train += e[2]
                total_td += td
                
    if total_td > 0:
        print(f"total avg {total_inf / total_td}, {total_train / total_td}")


if __name__ == "__main__":
    main()