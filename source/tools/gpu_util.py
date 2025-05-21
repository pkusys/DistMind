import sys
import datetime
from collections import OrderedDict

def import_log(filename):
    log = []
    with open(filename) as f:
        f.readline()
        for line in f.readlines():
            parts = line.split(',')
            timestamp = datetime.datetime.strptime(parts[0].strip(), '%Y/%m/%d %H:%M:%S.%f')
            index = int(parts[1].strip())
            utilization = float(parts[2].strip().strip('%'))
            log.append((timestamp, index, utilization))
    return log

def polish_log(log):
    # Time to seconds
    start_time = None
    for timestamp, _, _ in log:
        if start_time is None:
            start_time = timestamp
        elif timestamp < start_time:
            start_time = timestamp
    log = [((timestamp - start_time).total_seconds(), index, utilization) for (timestamp, index, utilization) in log]
    
    return log

def extract_gpus(log):
    gpus = set()
    for _, index, _ in log:
        gpus.add(index)
    return gpus

def extract_util_sequence(log, target):
    second_map = {}
    for timestamp, index, utilization in log:
        if index == target:
            second = int(timestamp)
            if second not in second_map:
                second_map[second] = []
            second_map[second].append(utilization)
    second_map = OrderedDict([(second, sum(util_list) / len(util_list)) for (second, util_list) in second_map.items()])
    return [util for _, util in second_map.items()]

def export_summary(filename, summary):
    with open(filename, 'w') as f:
        for timestamp, all_utilization in summary:
            all_utilization_str = [str(u) for u in all_utilization]
            f.write('%s, %s\n' % (timestamp, ', '.join(all_utilization_str)))

def main():
    input_filename = sys.argv[1]
    output_filename = sys.argv[2]

    # Import log
    log = import_log(input_filename)
    # Polish log
    log = polish_log(log)

    # Extract list of GPUs
    gpus = extract_gpus(log)

    # Extract utilization sequence for each GPU
    util_seqs = {}
    for gpu in gpus:
        util_seqs[gpu] = extract_util_sequence(log, gpu)
    length = 0
    for gpu in gpus:
        length = len(util_seqs[gpu])
        break
    for i in range(length):
        total = 0.0
        for gpu in gpus:
            total += util_seqs[gpu][i]
        print (total)

    # summary = []
    # for i in range(num_snapshot):
    #     timestamp = log[i * num_gpu][0]

    #     all_utilization = []
    #     for j in range(num_gpu):
    #         _, _, utilization = log[i * num_gpu + j]
    #         all_utilization.append(utilization)

    #     summary.append((timestamp, all_utilization))

    # export_summary(output_filename, summary)

if __name__ == "__main__":
    main()