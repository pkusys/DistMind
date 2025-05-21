import os
from collections import OrderedDict
import sys
import torch
import torch.nn.functional as F
import torch.distributed as dist
import model.common.aux_func as auxF 

from model.common.loss import create_criterion
from model.common.optimizer import create_optimizer

distributed_training = False

def initialize_distributed_training(backend, world_size, rank, master_addr, master_port):
    global distributed_training

    torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    distributed_training = True

def expand_module(mod, fullname):
    if (len(list(mod.children())) == 0):
        return [(mod, fullname)]
    else:
        layer_list = []
        for child_name, child in mod.named_children():
            child_fullname = fullname + "/" + child_name
            layer_list += expand_module(child, child_fullname)
        return layer_list


def extract_func_info(func_list):
    func_info_list = []
    param_list = []
    for layer_name, input_index, param, hyperparam, forward_pre_hooks, forward_hooks in func_list:
        param_info = []
        param_data = []
        ordered_param = OrderedDict(sorted(param.items()))
        for key, p in ordered_param.items():
            param_info.append((key, p.shape, p.dtype))
            param_data.append(p)
        func_info_list.append(
            (layer_name, input_index, param_info, hyperparam, forward_pre_hooks, forward_hooks))
        param_list.append(ordered_param)
    return func_info_list, param_list


def str2fn(func_name):
    try:
        func = getattr(F, func_name)
    except:
        try:
            func = getattr(torch, func_name)
        except:
            try:
               func = getattr(auxF, func_name)
            except:
                raise Exception("cannot find suitable func with name " + func_name)
    if func_name == "tanh":
        func = torch.tanh # pylint: disable=no-member
    return func

def evaluate_model_forward(input_batch, func_info, param_list):
    intermediate = [input_batch]
    for layer_info, param in zip(func_info, param_list):
        func_name, input_index, _, hyperparam, forward_pre_hooks, forward_hooks = layer_info
        func = str2fn(func_name)

        input = [intermediate[i] for i in input_index]
        for fn in forward_pre_hooks:
            fn(input)
        # output = func(*input, **dict(param), **hyperparam)
        try:
            output = func(*input, **dict(param), **hyperparam)
        except Exception as e:
            print(f"Error in layer {func_name}")
            if func_name == "relu":
                try:
                    hyperparam['inplace'] = False
                    output = func(*input, **dict(param), **hyperparam)
                except Exception as e:
                    raise e
            else:
                raise e
        for fn in forward_hooks:
            fn(output)
        intermediate.append(output)
    return intermediate[-1]

def infer_model(input_batch, func_info, param_list):
    with torch.no_grad():
        output = evaluate_model_forward(input_batch, func_info, param_list)
    return output

def train_model(
    global_batch_size, input_batch, targets, func_info, param_list,
    loss_name, optimizer_name, lr
):
    batch_size = input_batch.shape[0]
    
    # Create criterion
    criterion = create_criterion(loss_name, reduction='sum')
    if torch.cuda.is_available():
        criterion = criterion.cuda()
    # Create optimizer
    optimizer, trainable_param_list = create_optimizer('sgd', param_list, lr / global_batch_size)
    optimizer.zero_grad()
    # Forward
    output = evaluate_model_forward(input_batch, func_info, param_list)
    # Loss
    loss = criterion(output, targets)
    # Backward
    loss.backward()
    # All-Reduce
    # if distributed_training and batch_size < global_batch_size:
    #    for p in trainable_param_list:
    #        dist.all_reduce(p.grad, op=dist.ReduceOp.SUM)
    # Update
    optimizer.step()
    # Check the result
    return loss / batch_size

def _debug_fn(prefix):
    """"""
    def _ofn(output):
        print(prefix, output.sum().item())

    return _ofn
