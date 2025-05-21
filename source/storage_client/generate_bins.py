from torchvision import models
import sys
import struct
import os


def save_partitions(save_to, layers, partitions, p_name):
    """"""
    for bidx in range(len(partitions)):
        batch = partitions[bidx]

        with open(os.path.join(save_to, "{}-{}.bin".format(p_name, bidx)), "wb") as ofile:
            for idx in batch:
                for param in layers[idx].parameters():
                    n_param = param.cpu().detach().numpy()
                    param_arr = n_param.flatten().tolist()
                    # append to file
                    d = struct.pack("%sf" % len(param_arr), *param_arr)
                    ofile.write(d)


def get_layers(module, layers):
    childs = list(module.children())
    if len(childs) == 0:
        layers.append(module)
    else:
        for c in childs:
            get_layers(c, layers)


def get_layers_size(layers):
    sizes = []

    for l in layers:
        n = 0
        for p in l.parameters():
            t = 1
            for s in p.size():
                t *= s
            n += t
        sizes.append(n * 4)  # assume 4 bytes for each parameter
    return sizes


def save_entire_model(save_to, name, layers):
    with open(os.path.join(save_to, "{}-fullmodel.bin".format(name)), "wb") as ofile:
        for l in layers:
            for param in l.parameters():
                n_param = param.cpu().detach().numpy()
                param_arr = n_param.flatten().tolist()
                # append to file
                d = struct.pack("%sf" % len(param_arr), *param_arr)
                ofile.write(d)


def partition(sizes):
    """
    """
    partitions = []
    # first batch 6MB
    # the rest 22MB each
    first_batch = 10 * 1024 * 1024

    batch = []
    cidx = 0
    acc = 0
    while acc < first_batch and cidx < len(sizes):
        batch.append(cidx)

        acc += sizes[cidx]
        cidx += 1
    partitions.append(batch)

    other_batch = 32 * 1024 * 1024

    batch = []
    acc = 0
    for i in range(cidx, len(sizes)):
        if acc < other_batch:
            pass
        else:
            partitions.append(batch)
            batch = []
            acc = 0

        batch.append(i)
        acc += sizes[i]

    if len(batch) != 0:
        partitions.append(batch)

    return partitions


def main():
    """"""
    if len(sys.argv) < 2:
        print('require an output directory')
        return -1

    output_dir = sys.argv[1]
    model_name = "resnet152"
    model = models.resnet152(pretrained=True)
    
    layers = []
    get_layers(model, layers)
    save_entire_model(output_dir, model_name, layers)

    sizes = get_layers_size(layers)
    parts = partition(sizes)

    save_to = os.path.join(output_dir, 'layer_batches')
    if not os.path.exists(save_to):
        os.makedirs(save_to)
    save_partitions(save_to, layers, parts, model_name)


if __name__ == "__main__":
    main()
