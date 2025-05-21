import os
import sys

import torch

from model.resnet import resnet

# TODO: Adapt API

def _import_data(batch_size):
    return resnet.import_data(batch_size)

def _import_model():
    return resnet.import_model('resnet18')

def _import_model_reimpl():
    layer_list = resnet.import_layer_list('resnet18')
    partial_func_list = resnet._make_model(resnet._make_basicblock, [2, 2, 2, 2])
    return resnet._make_func_list(partial_func_list, layer_list)