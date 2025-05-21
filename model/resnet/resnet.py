import os
import sys

import torch
import torch.nn.functional as F
from torchvision.models import resnet152
from model.common.serialize import extract_hyperparameters

def import_data(batch_size):
    filename = 'dog.jpg'

    # Download an example image from the pytorch website
    if not os.path.isfile(filename):
        import urllib
        url = 'https://github.com/pytorch/hub/blob/master/images/dog.jpg?raw=true'
        try: 
            urllib.request.URLopener().retrieve(url, filename)
        except: 
            urllib.request.urlretrieve(url, filename)

    # sample execution (requires torchvision)
    from PIL import Image
    from torchvision import transforms
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    image = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    images = torch.cat([image] * batch_size)
    target = torch.tensor([0] * batch_size)
    return images, target

def import_model(model_name, train=False):
    if model_name == 'resnet152':
        model = resnet152(pretrained=True)
    else:
        model = torch.hub.load('pytorch/vision:release/0.14', model_name, pretrained=True)
    def set_mod_fullname(mod, fullname):
        mod.fullname = fullname
        for child_name, child in mod.named_children():
            child_fullname = fullname + "/" + child_name
            set_mod_fullname(child, child_fullname)
    set_mod_fullname(model, model_name)
    if train:
        model.train()
    else:
        model.eval()
    return model

def import_layer_list(model_name, train=False, device='cuda'):
    model = import_model(model_name, train)
    data, _ = import_data(8)
    layer_list = []
    def add_layer_to_list(mod, input, output):
        layer_list.append((mod, mod.fullname))
    def add_hook_for_extract_layers(mod):
        if (len(list(mod.children())) == 0):
            mod.register_forward_hook(add_layer_to_list)
        else:
            for child in mod.children():
                add_hook_for_extract_layers(child)
    add_hook_for_extract_layers(model)
    with torch.no_grad():
        model(data)
    return layer_list

def _make_downsample(input_index, base_index, norm_layer='batch_norm'):
    partial_func_list = []
    partial_func_list.append(('conv2d', [input_index]))
    partial_func_list.append((norm_layer, [base_index + len(partial_func_list)]))
    return partial_func_list

def _make_basicblock(base_index, norm_layer='batch_norm', downsample=False):
    partial_func_list = []
    partial_func_list.append(('conv2d', [base_index + len(partial_func_list)]))
    partial_func_list.append((norm_layer, [base_index + len(partial_func_list)]))
    partial_func_list.append(('relu', [base_index + len(partial_func_list)]))
    partial_func_list.append(('conv2d', [base_index + len(partial_func_list)]))
    partial_func_list.append((norm_layer, [base_index + len(partial_func_list)]))
    intermediate_index = base_index + len(partial_func_list)

    if downsample:
        partial_func_list += _make_downsample(base_index, base_index + len(partial_func_list), norm_layer)
        residual_index = base_index + len(partial_func_list)
    else:
        residual_index = base_index

    partial_func_list.append(('add', [intermediate_index, residual_index]))
    partial_func_list.append(('relu', [base_index + len(partial_func_list)]))
    return partial_func_list

def _make_bottleneck(base_index, norm_layer='batch_norm', downsample=False):
    partial_func_list = []
    partial_func_list.append(('conv2d', [base_index + len(partial_func_list)]))
    partial_func_list.append((norm_layer, [base_index + len(partial_func_list)]))
    partial_func_list.append(('relu', [base_index + len(partial_func_list)]))
    partial_func_list.append(('conv2d', [base_index + len(partial_func_list)]))
    partial_func_list.append((norm_layer, [base_index + len(partial_func_list)]))
    partial_func_list.append(('relu', [base_index + len(partial_func_list)]))
    partial_func_list.append(('conv2d', [base_index + len(partial_func_list)]))
    partial_func_list.append((norm_layer, [base_index + len(partial_func_list)]))
    intermediate_index = base_index + len(partial_func_list)

    if downsample:
        partial_func_list += _make_downsample(base_index, base_index + len(partial_func_list), norm_layer)
        residual_index = base_index + len(partial_func_list)
    else:
        residual_index = base_index

    partial_func_list.append(('add', [intermediate_index, residual_index]))
    partial_func_list.append(('relu', [base_index + len(partial_func_list)]))
    return partial_func_list

def _make_layer(base_index, make_block, blocks, norm_layer='batch_norm', downsample=True):
    partial_func_list = []
    partial_func_list += make_block(base_index + len(partial_func_list), norm_layer, downsample)
    for _ in range(1, blocks):
        partial_func_list += make_block(base_index + len(partial_func_list), norm_layer)
    return partial_func_list

def _make_model(make_block, layers, norm_layer='batch_norm', layer1_downsample=False):
    partial_func_list = []
    partial_func_list.append(('conv2d', [len(partial_func_list)]))
    partial_func_list.append((norm_layer, [len(partial_func_list)]))
    partial_func_list.append(('relu', [len(partial_func_list)]))
    partial_func_list.append(('max_pool2d', [len(partial_func_list)]))
    partial_func_list += _make_layer(len(partial_func_list), make_block, layers[0], norm_layer, layer1_downsample)
    partial_func_list += _make_layer(len(partial_func_list), make_block, layers[1], norm_layer, True)
    partial_func_list += _make_layer(len(partial_func_list), make_block, layers[2], norm_layer, True)
    partial_func_list += _make_layer(len(partial_func_list), make_block, layers[3], norm_layer, True)
    partial_func_list.append(('adaptive_avg_pool2d', [len(partial_func_list)]))
    partial_func_list.append(('flatten', [len(partial_func_list)]))
    partial_func_list.append(('linear', [len(partial_func_list)]))

    return partial_func_list

def _make_func_list(partial_func_list, layer_list):
    corresponding_layer_index = 0
    func_list = []
    for func_name, input_index in partial_func_list:
        if hasattr(F, func_name):
            layer = layer_list[corresponding_layer_index][0]
            corresponding_layer_index += 1
            params, hyperparams = extract_hyperparameters(layer, func_name)
            forward_pre_hooks = []
            forward_hooks = []
            func_list.append((func_name, input_index, params, hyperparams, forward_pre_hooks, forward_hooks))
        else:
            if func_name == 'flatten':
                func_list.append((func_name, input_index, {}, {'start_dim': 1}, [], []))
            elif func_name == 'add':
                func_list.append((func_name, input_index, {}, {}, [], []))
            else:
                print ('Error! Undefined func_name', func_name)
                sys.exit(0)
    return func_list