import sys
import random

def import_model_list_seed(filename):
    model_list_seed = []
    model_list_seed_weights = []
    with open(filename) as f:
        f.readline()
        for line in f.readlines():
            parts = line.split(',')
            model = parts[0].strip()
            weight = int(parts[1].strip())
            model_list_seed.append(model)
            model_list_seed_weights.append(weight)
    return model_list_seed, model_list_seed_weights

def main():
    model_list_seed_filename = sys.argv[1]
    num_models = int(sys.argv[2])
    rank = int(sys.argv[3])
    model_list_filename = sys.argv[4]

    random.seed(410)

    model_list_seed, model_list_seed_weights = import_model_list_seed(model_list_seed_filename)

    # model_list = ['%s-train' % seed for seed in model_list_seed]
    if rank == 1:
        model_list = ['resnet152_train']
    else:
        model_list = []

    model_list_raw = random.choices(model_list_seed, weights=model_list_seed_weights, k=num_models)
    model_list += ['%s-alter-%s-%08d' % (model, rank, i) for i, model in enumerate(model_list_raw)]

    with open(model_list_filename, 'w') as f:
        f.write('model_name\n')
        for model_name in model_list:
            f.write(model_name + '\n')

if __name__ == "__main__":
    main()