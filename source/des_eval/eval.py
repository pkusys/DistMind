from model.index import get_model_module
import pickle
import time
import torch
import numpy as np
from model.common.util import extract_func_info
import os
import argparse

def strip_all_parameters(model: torch.nn.Module):
    """"""
    for p in model.parameters():
        p.data = torch.tensor([0.0])
    return model

def eval_once(model_name):
    mod = get_model_module(model_name)

    model = mod.import_model()
    data, _ = mod.import_data(8)

    # make sure the model has been loaded
    # outputs = model(data)
    # print(model_name, 'outputs', outputs[0].sum().item())

    ret = []
    # pickle the entire model
    s = time.time()
    model_b = pickle.dumps(model)
    e = time.time()
    ret.append(e - s) # serialization time
    print(model_name, 'model byte sizes:', len(model_b))
    # unpickle entire model
    s = time.time()
    another_model = pickle.loads(model_b)
    e = time.time()
    ret.append(e - s) # deserialization time
    print(model_name, 'deserialize full model', e - s)

    # strip all parameters
    stripped_model = strip_all_parameters(model)
    s = time.time()
    stripped_b = pickle.dumps(stripped_model)
    e = time.time()
    ret.append(e - s)
    print(model_name, 'model size after strip all params', len(stripped_b))

    # deserialize the stripped model
    s = time.time()
    stripped_model = pickle.loads(stripped_b)
    e = time.time()
    ret.append(e - s)
    print(model_name, 'deserialize stripped model', e - s)

    func_list_model = mod.import_model_reimpl()
    func_list, model_params = extract_func_info(func_list_model)
    # serialization
    s = time.time()
    func_list_b = pickle.dumps(func_list)
    e = time.time()
    ret.append(e - s)
    # deserialization
    s = time.time()
    d_func_list = pickle.loads(func_list_b)
    e = time.time()
    ret.append(e - s)
    print(model_name, 'deserialize function list', e - s)

    del model
    del model_b
    del another_model
    del stripped_model
    del stripped_b
    del model_params
    del func_list
    del func_list_b
    del d_func_list
    return ret


def main():
    """"""
    models = ["resnet152", "inception_v3", "densenet201", "bert_base", "gpt2"]
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--output",
        type=str,
        default="eval_results.txt",
        help="output file to save the results",
    )
    args = argparser.parse_args()
    output = args.output

    with open(output, "w") as f:
        f.write("")
        for m in models:
            repeated = []
            for _ in range(10):
                res = eval_once(m)
                repeated.append(res)
            repeated = np.array(repeated)
            res = np.mean(repeated, axis=0).tolist()
            msg = "{} ser/des full model {} {} ser/des stripped model: {} {} ser/des func list {} {}".format(
                m, *res
            )
            print(">"*10, msg)
            f.write(msg + "\n")


if __name__ == "__main__":
    main()