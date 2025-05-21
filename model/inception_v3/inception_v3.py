import os
import torch
from torchvision import models

from model.common.batch import generate_batch_basic
from model.common.serialize import extract_hyperparameters

MODEL_NAME = 'inception_v3'

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
        transforms.Resize(320),
        transforms.CenterCrop(299),
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
    model = models.inception_v3(pretrained=True)
    def set_mod_fullname(mod, fullname):
        mod.fullname = fullname
        for child_name, child in mod.named_children():
            child_fullname = fullname + "/" + child_name
            set_mod_fullname(child, child_fullname)
    set_mod_fullname(model, "inception_v3")
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


def _make_basic_conv2d(func_list, layers, base_idx, input_idx):
    """"""
    func_list.append(("conv2d", [input_idx], *extract_hyperparameters(layers[base_idx][0], "conv2d"), [], []))
    func_list.append(("batch_norm", [-1], *extract_hyperparameters(layers[base_idx+1][0], "batch_norm"), [], []))
    func_list.append(("relu", [-1], {}, {"inplace": True}, [], []))


def _make_inceptionA(func_list, layers, base_idx):
    """ 24 outputs """
    _make_basic_conv2d(func_list, layers, base_idx, -1) # self.branch1x1(x)
    _make_basic_conv2d(func_list, layers, base_idx + 2, -4) # self.branch5x5_1(x)
    _make_basic_conv2d(func_list, layers, base_idx + 2 * 2, -1) # self.branch5x5_2(branch5x5)
    _make_basic_conv2d(func_list, layers, base_idx + 2 * 3, -10) # self.branch3x3dbl_1(x)
    _make_basic_conv2d(func_list, layers, base_idx + 2 * 4, -1) # self.branch3x3dbl_2(branch3x3dbl)
    _make_basic_conv2d(func_list, layers, base_idx + 2 * 5, -1) # self.branch3x3dbl_3(branch3x3dbl)

    func_list.append(("avg_pool2d", [-19], {}, {"kernel_size":3, "stride":1, "padding":1}, [], [])) 
    _make_basic_conv2d(func_list, layers, base_idx + 2 * 6, -1) # self.branch_pool(branch_pool)
    func_list.append(("make_list", [-20, -14, -5,-1], {}, {}, [], []))
    func_list.append(("cat", [-1], {}, {"dim":1}, [], []))


def _make_inceptionB(func_list, layers, base_idx):
    """ 15 outputs """
    _make_basic_conv2d(func_list, layers, base_idx, -1) # self.branch3x3(x) 

    _make_basic_conv2d(func_list, layers, base_idx + 2, -4) # self.branch3x3dbl_1(x) 
    _make_basic_conv2d(func_list, layers, base_idx + 2*2, -1) # self.branch3x3dbl_2(branch3x3dbl) 
    _make_basic_conv2d(func_list, layers, base_idx + 2*3, -1) # self.branch3x3dbl_3(branch3x3dbl) 

    func_list.append(("max_pool2d", [-13], {}, {"kernel_size":3, "stride":2}, [], [])) # F.max_pool2d(x, kernel_size=3, stride=2) -1
    func_list.append(("make_list", [-11,-2,-1], {}, {}, [],[])) # outputs = [branch3x3, branch3x3dbl, branch_pool]
    func_list.append(("cat", [-1], {}, {"dim":1}, [], [])) # return torch.cat(outputs, 1)

def _make_inceptionC(func_list, layers, base_idx):
    """33 outputs"""
    _make_basic_conv2d(func_list, layers, base_idx, -1) # self.branch1x1(x)

    _make_basic_conv2d(func_list, layers, base_idx+2, -4) # self.branch7x7_1(x) 
    _make_basic_conv2d(func_list, layers, base_idx+2*2, -1) # self.branch7x7_2(branch7x7) 
    _make_basic_conv2d(func_list, layers, base_idx+2*3, -1) # self.branch7x7_3(branch7x7) 

    _make_basic_conv2d(func_list, layers, base_idx+2*4, -13) # self.branch7x7dbl_1(x) 
    _make_basic_conv2d(func_list, layers, base_idx+2*5, -1) # self.branch7x7dbl_2(branch7x7dbl) 
    _make_basic_conv2d(func_list, layers, base_idx+2*6, -1) # self.branch7x7dbl_3(branch7x7dbl) 
    _make_basic_conv2d(func_list, layers, base_idx+2*7, -1) # self.branch7x7dbl_4(branch7x7dbl) 
    _make_basic_conv2d(func_list, layers, base_idx+2*8, -1) # self.branch7x7dbl_5(branch7x7dbl) 

    func_list.append(("avg_pool2d", [-28], {}, {"kernel_size":3, "stride":1, "padding":1}, [], [])) # F.avg_pool2d(x, kernel_size=3, stride=1, padding=1) -3
    _make_basic_conv2d(func_list, layers, base_idx+2*9, -1) # self.branch_pool(branch_pool) -1,-2
    func_list.append(("make_list", [-29, -20, -5, -1], {}, {}, [],[]))

    func_list.append(("cat", [-1], {}, {"dim":1}, [], []))


def _make_inceptionD(func_list, layers, base_idx):
    """ 21 outputs """
    _make_basic_conv2d(func_list, layers, base_idx, -1) # self.branch3x3_1(x)
    _make_basic_conv2d(func_list, layers, base_idx+2, -1) # self.branch3x3_2(branch3x3)

    _make_basic_conv2d(func_list, layers, base_idx+2*2, -7) # self.branch7x7x3_1(x)
    _make_basic_conv2d(func_list, layers, base_idx+2*3, -1) # self.branch7x7x3_2(branch7x7x3)
    _make_basic_conv2d(func_list, layers, base_idx+2*4, -1) # self.branch7x7x3_3(branch7x7x3)
    _make_basic_conv2d(func_list, layers, base_idx+2*5, -1) # self.branch7x7x3_4(branch7x7x3) 

    func_list.append(("max_pool2d", [-19], {}, {"kernel_size": 3, "stride": 2}, [], [])) # F.max_pool2d -1
    func_list.append(("make_list", [-14, -2,-1], {}, {}, [], []))
    func_list.append(("cat", [-1], {}, {"dim":1}, [], []))


def _make_inceptionE(func_list, layers, base_idx):
    """ 27 + 7 = 34"""
    _make_basic_conv2d(func_list, layers, base_idx, -1) # self.branch1x1(x) -30

    _make_basic_conv2d(func_list, layers, base_idx + 2, -4) # self.branch3x3_1(x) -27
    _make_basic_conv2d(func_list, layers, base_idx + 2*2, -1) # self.branch3x3_2a(branch3x3) -24
    _make_basic_conv2d(func_list, layers, base_idx + 2*3, -4) # self.branch3x3_2b(branch3x3) -21
    func_list.append(("make_list", [-4, -1], {}, {}, [], [])) # -20 
    func_list.append(("cat", [-1], {}, {"dim":1}, [], [])) # -19

    _make_basic_conv2d(func_list, layers, base_idx + 2*4, -15) # self.branch3x3dbl_1(x) -16
    _make_basic_conv2d(func_list, layers, base_idx + 2*5, -1) # self.branch3x3dbl_2(branch3x3dbl) -13
    # branch3x3dbl = [ self.branch3x3dbl_3a(branch3x3dbl), self.branch3x3dbl_3b(branch3x3dbl),]
    _make_basic_conv2d(func_list, layers, base_idx + 2 * 6, -1) # self.branch3x3dbl_3a(branch3x3dbl), -10
    _make_basic_conv2d(func_list, layers, base_idx + 2 * 7, -4) # self.branch3x3dbl_3b(branch3x3dbl) -7
    func_list.append(("make_list", [-4, -1], {}, {}, [], [])) # -6
    func_list.append(("cat", [-1], {}, {"dim":1}, [], [])) # -5

    func_list.append(("avg_pool2d", [-29], {}, {"kernel_size":3, "stride":1, "padding":1}, [], [])) # -4
    _make_basic_conv2d(func_list, layers, base_idx + 2 * 8, -1) # self.branch_pool(branch_pool) -1

    func_list.append(("make_list", [-30, -19, -5,-1], {}, {}, [], []))
    func_list.append(("cat", [-1], {}, {"dim":1}, [], []))


def _make_func_list(layers):
    """"""
    func_list = []

    func_list.append(("inception_transform_input", [0], {}, {}, [], [])) # self._transform_input(x) 1

    # _forward 
    _make_basic_conv2d(func_list, layers, 0, -1) # self.Conv2d_1a_3x3(x) 4
    _make_basic_conv2d(func_list, layers, 2, -1) # self.Conv2d_2a_3x3(x) 7
    _make_basic_conv2d(func_list, layers, 4, -1) # self.Conv2d_2b_3x3(x) 10

    func_list.append(("max_pool2d", [-1], {}, {"kernel_size":3, "stride":2}, [], [])) # 11
    _make_basic_conv2d(func_list, layers, 7, -1) # x = self.Conv2d_3b_1x1(x) # 14
    _make_basic_conv2d(func_list, layers, 9, -1) # self.Conv2d_4a_3x3(x) # 17
    func_list.append(("max_pool2d", [-1], {}, {"kernel_size":3, "stride":2}, [], [])) #18

    _make_inceptionA(func_list, layers, 12) # self.Mixed_5b(x) 42
    _make_inceptionA(func_list, layers, 26) # self.Mixed_5c(x) 66
    _make_inceptionA(func_list, layers, 40) # self.Mixed_5d(x) 90
    _make_inceptionB(func_list, layers, 54) # self.Mixed_6a(x) 105
    _make_inceptionC(func_list, layers, 62) # self.Mixed_6b(x) 138
    _make_inceptionC(func_list, layers, 82) # self.Mixed_6c(x) 171 v
    _make_inceptionC(func_list, layers, 102) # self.Mixed_6d(x) 204
    _make_inceptionC(func_list, layers, 122) # self.Mixed_6e(x) 237

    # aux is None in eval model
    _make_inceptionD(func_list, layers, 142) # self.Mixed_7a(x) 258
    _make_inceptionE(func_list, layers, 154) # self.Mixed_7b(x) 292
    _make_inceptionE(func_list, layers, 172) # self.Mixed_7c(x) 326

    func_list.append(("adaptive_avg_pool2d", [-1], {}, {"output_size": (1, 1)}, [], [])) # F.adaptive_avg_pool2d(x, (1, 1))
    func_list.append(("dropout", [-1], {}, {"training":False}, [], [])) # F.dropout(x, training=self.training)
    func_list.append(("flatten", [-1], {}, {"start_dim": 1}, [], [])) # torch.flatten(x, 1)

    func_list.append(("linear", [-1], *extract_hyperparameters(layers[-1][0], "linear"), [], [])) # x = self.fc(x)

    return func_list


def import_model_reimpl(train=False, device='cuda'):
    """"""
    layer_list = import_layer_list(train)
    func_list = _make_func_list(layer_list)
    return func_list

def import_model_reimpl_with_batching(train=False, device='cuda', max_batch_size=8 * 4096000):
    func_list = import_model_reimpl(train, device)
    return func_list, generate_batch_basic(func_list, max_batch_size=max_batch_size)