import os

import torch

from model.common.util import expand_module
from model.common.serialize import extract_hyperparameters

# TODO: Adapt API

def _import_data(batch_size):
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

def _import_model():
    model = torch.hub.load('pytorch/vision:release/0.14', 'alexnet', pretrained=True)
    model.eval()
    return model

def _import_model_reimpl():
    model = import_model()
    layer_list = expand_module(model, 'alexnet')

    func_list = [
        ('conv2d', [0], *extract_hyperparameters(layer_list[0][0], 'conv2d')),
        ('relu', [1], *extract_hyperparameters(layer_list[1][0], 'relu')),
        ('max_pool2d', [2], *extract_hyperparameters(layer_list[2][0], 'max_pool2d')),
        ('conv2d', [3], *extract_hyperparameters(layer_list[3][0], 'conv2d')),
        ('relu', [4], *extract_hyperparameters(layer_list[4][0], 'relu')),
        ('max_pool2d', [5], *extract_hyperparameters(layer_list[5][0], 'max_pool2d')),
        ('conv2d', [6], *extract_hyperparameters(layer_list[6][0], 'conv2d')),
        ('relu', [7], *extract_hyperparameters(layer_list[7][0], 'relu')),
        ('conv2d', [8], *extract_hyperparameters(layer_list[8][0], 'conv2d')),
        ('relu', [9], *extract_hyperparameters(layer_list[9][0], 'relu')),
        ('conv2d', [10], *extract_hyperparameters(layer_list[10][0], 'conv2d')),
        ('relu', [11], *extract_hyperparameters(layer_list[11][0], 'relu')),
        ('max_pool2d', [12], *extract_hyperparameters(layer_list[12][0], 'max_pool2d')),
        ('adaptive_avg_pool2d', [13], *extract_hyperparameters(layer_list[13][0], 'adaptive_avg_pool2d')),
        ('flatten', [14], [], {'start_dim': 1}),
        ('dropout', [15], *extract_hyperparameters(layer_list[14][0], 'dropout')),
        ('linear', [16], *extract_hyperparameters(layer_list[15][0], 'linear')),
        ('relu', [17], *extract_hyperparameters(layer_list[16][0], 'relu')),
        ('dropout', [18], *extract_hyperparameters(layer_list[17][0], 'dropout')),
        ('linear', [19], *extract_hyperparameters(layer_list[18][0], 'linear')),
        ('relu', [20], *extract_hyperparameters(layer_list[19][0], 'relu')),
        ('linear', [21], *extract_hyperparameters(layer_list[20][0], 'linear')),
    ]

    return func_list