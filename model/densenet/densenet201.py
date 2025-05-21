""" densenet201
"""
import os
import torch
from torchvision import models
from model.common.batch import generate_batch_basic
from model.common.serialize import extract_hyperparameters

MODEL_NAME = 'densenet201'

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

    images = torch.cat([image] * batch_size) # pylint: disable=no-member
    target = torch.tensor([0] * batch_size) # pylint: disable=not-callable
    return images, target


def import_model(train=False):
    """"""
    model = models.densenet201(pretrained=True)
    def set_mod_fullname(mod, fullname):
        mod.fullname = fullname
        for child_name, child in mod.named_children():
            child_fullname = fullname + "/" + child_name
            set_mod_fullname(child, child_fullname)
    set_mod_fullname(model, "densenet201")
    if train:
        model.train()
    else:
        model.eval()
    return model


def import_layer_list(train=False):
    """"""
    model = import_model(train)
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


def _make_denselayer(func_list, layers, base_idx):
    """ 7 intermediate results """
    func_list.append(("cat", [-1], {}, {"dim":1}, [], [])) # cat feature list
    # norm1
    func_list.append(("batch_norm", [-1], *extract_hyperparameters(layers[base_idx][0], "batch_norm"), [],[]))
    #relu1
    func_list.append(("relu", [-1], *extract_hyperparameters(layers[base_idx+1][0], "relu"), [], []))
    # conv1
    func_list.append(("conv2d", [-1], *extract_hyperparameters(layers[base_idx+2][0], "conv2d"), [], []))
    # norm2
    func_list.append(("batch_norm", [-1], *extract_hyperparameters(layers[base_idx+3][0], "batch_norm"), [], []))
    # relu2
    func_list.append(("relu", [-1], *extract_hyperparameters(layers[base_idx+4][0], "relu"), [], []))
    # conv2
    func_list.append(("conv2d", [-1], *extract_hyperparameters(layers[base_idx+5][0], "conv2d"), [], []))


def _make_transition(func_list, layers, base_idx):
    """"""
    # norm
    func_list.append(("batch_norm", [-1], *extract_hyperparameters(layers[base_idx][0], "batch_norm"), [], []))
    # relu
    func_list.append(("relu", [-1], *extract_hyperparameters(layers[base_idx+1][0], "relu"), [], []))
    # conv
    func_list.append(("conv2d", [-1], *extract_hyperparameters(layers[base_idx+2][0], "conv2d"), [], []))
    # pool
    func_list.append(("avg_pool2d", [-1], {}, {"kernel_size":2, "stride":2}, [], []))


def _make_denseblock(func_list, layers, base_idx, num_layers):
    """ 
    """
    n_layer_outputs = 7
    # features = [init_features]
    func_list.append(("make_list", [-1], {}, {},[],[]))
    for i in range(num_layers):
        _idx = base_idx + i * 6
        # new_features = layer(features)
        _make_denselayer(func_list, layers, _idx)
        # features.append(new_features)
        func_list.append(("list_append", [-(1+n_layer_outputs), -1], {}, {}, [], []))
    # torch.cat(features, 1)
    func_list.append(("cat", [-1], {}, {"dim":1}, [], []))


def _make_first_conv(func_list, layers):
    """ first conv block
    """
    # conv0, norm0, relu0, pool0
    func_list.append(("conv2d", [-1], *extract_hyperparameters(layers[0][0], "conv2d"), [], []))
    func_list.append(("batch_norm", [-1], *extract_hyperparameters(layers[1][0], "batch_norm"), [], []))
    func_list.append(("relu", [-1], *extract_hyperparameters(layers[2][0], "relu"), [], []))
    func_list.append(("max_pool2d", [-1], *extract_hyperparameters(layers[3][0], "max_pool2d"), [], []))


def _make_func_list(layers):
    """"""
    func_list = []
    block_config = (6, 12, 48, 32) # which is fixed for densenet201
    _make_first_conv(func_list, layers)

    base_idx = 4
    for i, n in enumerate(block_config):
        """"""
        _make_denseblock(func_list, layers, base_idx, n)
        base_idx += n * 6
        if i != len(block_config) - 1:
            _make_transition(func_list, layers, base_idx)
            base_idx += 4
    # norm5
    func_list.append(("batch_norm", [-1], *extract_hyperparameters(layers[base_idx][0], "batch_norm"), [], []))
    # ----------- end self.features

    func_list.append(("relu", [-1], {}, {"inplace": True}, [], []))
    func_list.append(("adaptive_avg_pool2d", [-1], {}, {"output_size": (1, 1)}, [], []))
    func_list.append(("flatten", [-1], {}, {"start_dim": 1}, [],[]))
    func_list.append(("linear",[-1], *extract_hyperparameters(layers[-1][0], "linear"), [], []))

    return func_list


def import_model_reimpl(train=False, device='cuda'):
    """"""
    layer_list = import_layer_list(train)
    func_list = _make_func_list(layer_list)
    return func_list

def import_model_reimpl_with_batching(train=False, device='cuda', max_batch_size=8 * 4096000):
    func_list = import_model_reimpl(train, device)
    return func_list, generate_batch_basic(func_list, max_batch_size=max_batch_size)