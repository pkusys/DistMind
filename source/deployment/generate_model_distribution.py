import sys
import random

GLOBAL_DISTRIBUTION_THRESHOLD = 4

def count_storage(filename):
    with open(filename) as f:
        return len(f.readlines()) - 2

def import_model_list(filename):
    model_list = []
    with open(filename) as f:
        f.readline()
        for line in f.readlines():
            parts = line.split(',')
            model = parts[0].strip()
            model_list.append(model)
    return model_list

def main():
    storage_list_filename = sys.argv[1]
    model_list_filename = sys.argv[2]
    model_distribution_filename = sys.argv[3]

    random.seed(410)

    total_storage = count_storage(storage_list_filename)
    model_list = import_model_list(model_list_filename)
    
    model_distribution = []
    distribution_threshold = min(total_storage, GLOBAL_DISTRIBUTION_THRESHOLD)
    for model_name in model_list:
        dist_num = distribution_threshold
        distribution = random.sample(list(range(total_storage)), dist_num)
        model_distribution.append((model_name, distribution))
    
    with open(model_distribution_filename, 'w') as f:
        f.write('model_name, storage_list\n')
        for model_name, distribution in model_distribution:
            distribution_str = [str(d) for d in distribution]
            f.write(', '.join([model_name] + distribution_str) + '\n')

if __name__ == "__main__":
    main()