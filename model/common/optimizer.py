import torch

def create_optimizer(optimizer_name, param_list, lr):
    trainable_param_list = []
    for param in param_list:
        for key, p in param:
            if key != 'running_mean' and key != 'running_var':
                p.requires_grad = True
                trainable_param_list.append(p)

    if optimizer_name == "sgd":
        return torch.optim.SGD(trainable_param_list, lr), trainable_param_list
    else:
        return None, None