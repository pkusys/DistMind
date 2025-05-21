import os
import sys

import torch

from model.common.batch import generate_batch_basic
from model.resnet import resnet


MODEL_NAME = 'resnet152'

def import_data(batch_size):
    return resnet.import_data(batch_size)

def import_model(train=False):
    return resnet.import_model('resnet152', train)

def import_model_reimpl(train=False, device='cuda'):
    layer_list = resnet.import_layer_list('resnet152', train, device)
    partial_func_list = resnet._make_model(resnet._make_bottleneck, [3, 8, 36, 3], layer1_downsample=True)
    return resnet._make_func_list(partial_func_list, layer_list)

def import_model_reimpl_with_batching(train=False, device='cuda', max_batch_size=8 * 4096000):
    func_list = import_model_reimpl(train, device)
    return func_list, generate_batch_basic(func_list, max_batch_size=max_batch_size)