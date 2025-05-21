from transformers import GPT2Tokenizer, GPT2Model
import torch
from model.common.batch import generate_batch_basic
from model.common.serialize import extract_hyperparameters
from model.common.util import _debug_fn

MODEL_NAME = 'gpt2'

def outputs_hook_fn(module, input, output):
    print(module.fullname, "::", output.sum().item())

def layerwise_outputs_hooks(module):
    childs = list(module.children())
    if len(childs) == 0:
        module.register_forward_hook(outputs_hook_fn)
    else:
        for c in childs:
            layerwise_outputs_hooks(c)

def import_data(batch_size):
    """"""
    test_inputs = ["This document provides solutions to a variety of use cases regarding the saving and loading of PyTorch models.",
                   " Feel free to read the whole document, or just skip to the code you need for a desired use case.",
                   "When it comes to saving and loading models, there are three core functions to be familiar with:",
                   " Saves a serialized object to disk. ",
                   "This function uses Python’s pickle utility for serialization.",
                   "Models, tensors, and dictionaries of all kinds of objects can be saved using this function.",
                   "Uses pickle’s unpickling facilities to deserialize pickled object files to memory.",
                   "This function also facilitates the device to load the data into."]

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = "<END>"

    input_ids_seed = []
    for sent in test_inputs:
        _ids = tokenizer.encode(
            sent, max_length=512, padding='max_length', truncation=True)
        input_ids_seed.append(_ids)
    
    input_ids = []
    while len(input_ids) < batch_size:
        input_ids += input_ids_seed
    input_ids = input_ids[:batch_size]
    
    # pylint: disable=not-callable
    input_ids = torch.tensor(input_ids)
    target = torch.tensor([0] * len(test_inputs))  # fake
    return input_ids, target

def import_model(train=False):
    """ BertModel
    only used for extract model structure for inference
    """
    model = GPT2Model.from_pretrained('gpt2')

    # model = torch.hub.load('pytorch/vision:release/0.14', model_name, pretrained=True)
    def set_mod_fullname(mod, fullname):
        mod.fullname = fullname
        for child_name, child in mod.named_children():
            child_fullname = fullname + "/" + child_name
            set_mod_fullname(child, child_fullname)

    set_mod_fullname(model, 'gpt2')
    if train:
        model.train()
    else:
        model.eval()
    return model

def import_layer_list(train=False):
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

    print("Hooking dropout")
    dropout_list = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Dropout) and 'attn.attn_dropout' in name:
            dropout_list.append(module)
    print("Hooking dropout :", len(dropout_list))

    new_layer_list = []
    dropout_index = 0
    i = 0
    while i < len(layer_list):
        mod, name = layer_list[i]
        new_layer_list.append((mod, name))

        # 当前与下一个均为 Conv1D，插入 Dropout
        if "c_attn" in name and i + 1 < len(layer_list):
            next_mod, next_name = layer_list[i + 1]
            if "c_proj" in next_name:
                if dropout_index < len(dropout_list):
                    dropout_mod = dropout_list[dropout_index]
                    dropout_name = dropout_mod.fullname
                    new_layer_list.append((dropout_mod, dropout_name))
                    dropout_index += 1
        i += 1

    return new_layer_list


def _make_MLP(func_list, layers, base_idx):
    """ 4 intermediate outputs """
    # self.c_fc(x)
    func_list.append(
        ("gpt2_conv1d", [-1], 
        *extract_hyperparameters(layers[base_idx][0], "gpt2_conv1d"), [], [])
    )
    # self.act(
    func_list.append(
        ("gpt2_gelu_new", [-1], {}, {}, [], [])
    )
    # self.c_proj(h)
    func_list.append(
        ("gpt2_conv1d", [-1], 
        *extract_hyperparameters(layers[base_idx+2][0], "gpt2_conv1d"), [], [])
    )
    # self.dropout(h2)
    func_list.append(
        ("dropout", [-1], *extract_hyperparameters(layers[base_idx+3][0], "dropout"), [], [])
    )

def _make_Attention(func_list, layers, base_idx, device):
    """ 18 intermediate outputs
     self.split_size = 768 """
    # self.c_attn(x)
    func_list.append(
        ('gpt2_conv1d', [-1], 
            *extract_hyperparameters(layers[base_idx][0], "gpt2_conv1d"),
            [], [])
    )
    # query, key, value = x.split(self.split_size, dim=2)
    func_list.append(
        ('split', [-1], {}, {'split_size_or_sections':768, 'dim':2}, [], [])
    )
    # 
    func_list.append(('gpt2_split_heads', [-1], {}, {'idx':0}, [], [])) # query = self.split_heads(query)
    func_list.append(('gpt2_split_heads', [-2], {}, {'idx':1, 'k':True}, [], [])) # key = self.split_heads(key, k=True)
    func_list.append(('gpt2_split_heads', [-3], {}, {'idx':2}, [], [])) # value = self.split_heads(value)

    # present = torch.stack((key.transpose(-2, -1), value))
    func_list.append(('transpose', [-2], {}, {'dim0': -2, 'dim1': -1}, [], []))
    func_list.append(('make_list', [-1, -2], {}, {}, [], []))
    func_list.append(('stack', [-1], {}, {}, [], []))

    # start self._attn(query, key, value ->
    # {w = w * b - 1e4 * (1 - b)}
    func_list.append(('gpt2_attn_w', [-6, -5, -4], {}, {'device': device}, [], []))
    func_list.append(('softmax', [-1], {}, {'dim':-1}, [], []))
    # w = self.attn_dropout(w)
    func_list.append(('dropout', [-1], *extract_hyperparameters(layers[base_idx+1][0], 'dropout'), [], []))
    # outputs = [torch.matmul(w, v)]
    func_list.append(('matmul', [-1, -7], {}, {}, [], []))
    func_list.append(('make_list', [-1], {}, {}, [], []))
    # end self._attn

    # a = attn_outputs[0]
    func_list.append(('get_ith', [-1], {}, {'i': 0}, [], []))
    # a = self.merge_heads(a)
    func_list.append(('gpt2_merge_head', [-1], {}, {}, [], []))
    # a = self.c_proj(a)
    func_list.append(('gpt2_conv1d', [-1], *extract_hyperparameters(layers[base_idx+2][0], "gpt2_conv1d"), [], []))
    # a = self.resid_dropout(a)
    func_list.append(('dropout', [-1], *extract_hyperparameters(layers[base_idx+3][0], "dropout"), [], []))
    # outputs = [a, present] + attn_outputs[1:] # (attn_outputs[1:] is None)
    func_list.append(('make_list', [-1, -10], {}, {}, [], []))


def _make_Block(func_list, layers, base_idx, device):
    """ 29 intermediate outputs """
    # self.ln_1(x)
    func_list.append(('layer_norm', [-1], *extract_hyperparameters(layers[base_idx][0], "layer_norm"), [], []))
    _make_Attention(func_list, layers, base_idx+1, device)  # 18 outputs
    # a = output_attn[0]
    func_list.append(('get_ith', [-1], {}, {'i':0}, [], []))
    # x = x + a
    func_list.append(('gpt2_add', [-1, -21], {}, {}, [], []))
    # self.ln_2(x)
    func_list.append(('layer_norm', [-1], *extract_hyperparameters(layers[base_idx+5][0], 'layer_norm'), [], []))
    # self.mlp(
    _make_MLP(func_list, layers, base_idx + 6) # 4 outputs
    # x = x + m
    func_list.append(('gpt2_add', [-6, -1], {}, {}, [], []))
    # outputs = [x] + output_attn[1:] # output_attn has only 2 elements
    func_list.append(('get_ith', [-9], {}, {'i':1}, [], []))
    func_list.append(('make_list', [-2, -1], {}, {}, [], []))


def _gpt2_prep(func_list, layers, device):
    """ get the inputs, go through wte, wpe etc """
    # input_shape = input_ids.size()
    func_list.append(('input_size', [0], {}, {}, [], []))
    func_list.append(('put_val', [], {}, {'v':-1}, [], []))
    func_list.append(('get_ith', [1], {}, {'i': -1}, [], []))
    # input_ids = input_ids.view(-1, input_shape[-1])
    func_list.append(('tensor_view', [0, -2, -1], {}, {}, [], []))
    # get new shape
    func_list.append(('tensor_shape', [-1], {}, {}, [], []))
    # batch_size = input_ids.shape[0]
    func_list.append(('get_ith', [-1], {}, {'i':0}, [], []))

    # position_ids
    func_list.append(('arange', [-4], {}, {'dtype': torch.long, 'device': device}, [], [])) # pylint: disable=no-member
    func_list.append(('unsqueeze', [-1], {}, {'dim':0}, [], []))
    func_list.append(('tensor_view', [-1, -7, -6], {}, {}, [], []))

    # inputs_embeds = self.wte(input_ids)
    func_list.append(('embedding', [-6], *extract_hyperparameters(layers[0][0], 'embedding'), [], []))
    # position_embeds = self.wpe(position_ids)
    func_list.append(('embedding', [-2], *extract_hyperparameters(layers[1][0], 'embedding'), [], []))
    # hidden_states = inputs_embeds + position_embeds + token_type_embeds # token_type_embeds == 0
    func_list.append(('gpt2_add', [-2, -1], {}, {}, [], []))
    # self.drop(hidden_states)
    func_list.append(('dropout', [-1], *extract_hyperparameters(layers[2][0], 'dropout'), [], []))

    # output_shape = input_shape + (hidden_states.size(-1),)
    func_list.append(('gpt2_output_shape', [1, -1], {}, {}, [], []))


def _make_func_list(layers, device='cuda'):
    """ hidden blocks: 12 """
    func_list = []

    _gpt2_prep(func_list, layers, device)
    # init presents = []
    func_list.append(('empty_list', [], {}, {}, [], []))
    # put hidden status as last item
    func_list.append(('put_val', [-3], {}, {}, [], []))
    base_idx = 3
    for _ in range(12):
        _make_Block(func_list, layers, base_idx, device) # 29 outputs
        base_idx += 10
        # present = outputs[1]
        func_list.append(('get_ith', [-1], {}, {'i':1}, [], []))
        func_list.append(('list_append', [-32, -1], {}, {}, [], []))
        # get hidden_states
        func_list.append(('get_ith', [-3], {}, {'i':0}, [], []))
    
    #  self.ln_f(hidden_states)
    func_list.append(('layer_norm', [-1], *extract_hyperparameters(layers[base_idx][0], 'layer_norm'), [], []))

    # make outputs
    # func_list.append(('make_list', [-1, -3], {}, {}, [], []))

    return func_list


def import_model_reimpl(train=False, device='cuda'):
    layers = import_layer_list(train)
    func_list = _make_func_list(layers, device)
    return func_list

def import_model_reimpl_with_batching(train=False, device='cuda', max_batch_size=8 * 4096000):
    func_list = import_model_reimpl(train, device)
    return func_list, generate_batch_basic(func_list, max_batch_size=max_batch_size)
