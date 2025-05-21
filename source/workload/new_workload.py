import sys

import numpy as np

def import_model_list(filename):
    model_list = []
    with open(filename) as f:
        _ = f.readline()
        for line in f.readlines():
            parts = line.split(',')
            model_name = parts[0].strip()
            if 'train' not in model_name:
                model_list.append(model_name)
    return model_list

def generate_zipf_distribution(n, s):
    weights = np.power(1.0 / np.arange(1, n + 1), s)
    distribution = weights / np.sum(weights)
    return list(distribution)

def generate_request_model(model_list, n, k, loop, zipf_s):
    if loop == 'loop':
        num_models = len(model_list)
        batch_list = [model_list[i % num_models] for i in range(n)]
    else:
        distribution = generate_zipf_distribution(len(model_list), zipf_s)
        batch_list = list(np.random.choice(model_list, n, p=distribution))

    request_list = [batch_list[i // k] for i in range(n * k)]
    return request_list

def generate_arrival_interval(n, rate, uniform):
    if uniform == 'uniform':
        return [1.0 / rate] * n
    else:
        return list(np.random.exponential(1.0 / rate, [n]))

def export_request_list(filename, request_model_list, request_interval_list):
    with open(filename, 'w') as f:
        f.write('model_name, batch_size, send_interval\n')
        for req_model, req_interval in zip(request_model_list, request_interval_list):
            f.write('%s, %d, %f\n' % (req_model, 8, req_interval))

def main():
    model_list_filename = sys.argv[1]
    output_filename = sys.argv[2]
    num_requests = int(sys.argv[3])
    batch_size = int(sys.argv[4])
    request_loop = sys.argv[5]
    zipf_s = float(sys.argv[6])
    throughput = float(sys.argv[7])
    uniform_interval = sys.argv[8]    

    num_batch = (num_requests + batch_size - 1) // batch_size
    num_requests = num_batch * batch_size

    model_list = import_model_list(model_list_filename)

    request_model_list = generate_request_model(model_list, num_batch, batch_size, request_loop, zipf_s)
    request_interval_list = generate_arrival_interval(num_requests, throughput, uniform_interval)

    export_request_list(output_filename, request_model_list, request_interval_list)

if __name__ == "__main__":
    main()