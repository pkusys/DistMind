from transformers import BertTokenizer, BertModel
import torch
from model.common.batch import generate_batch_basic
from model.common.serialize import extract_hyperparameters
import os

MODEL_NAME = 'bert_base'

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

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
    model = BertModel.from_pretrained('bert-base-uncased')
    def set_mod_fullname(mod, fullname):
        mod.fullname = fullname
        for child_name, child in mod.named_children():
            child_fullname = fullname + "/" + child_name
            set_mod_fullname(child, child_fullname)
    set_mod_fullname(model, 'bert_base')
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
        if isinstance(module, torch.nn.Dropout) and 'attention.self' in name:
            dropout_list.append(module)
    
    new_layer_list = []
    linear_buffer = []
    dropout_index = 0

    for mod, name in layer_list:
        new_layer_list.append((mod, name))

        if isinstance(mod, torch.nn.Linear):
            linear_buffer.append((mod, name))
            if len(linear_buffer) == 3:
                # 插入 dropout
                if dropout_index < len(dropout_list):
                    dropout_mod = dropout_list[dropout_index]
                    dropout_name = dropout_mod.fullname
                    new_layer_list.append((dropout_mod, dropout_name))
                    dropout_index += 1
                linear_buffer = []  # reset
        else:
            linear_buffer = []  # reset if not linear

    return new_layer_list


def _bert_prep(func_list, layers, device):
    """"""
    # input shape, add element to intermediate results
    func_list.append(("input_size", [0], {}, {}, [], [])) # 1
    # attention mask
    func_list.append(("ones", [1], {}, {'device': device}, [], [])) # 2
    # token_type_ids
    func_list.append(("zeros", [1], {}, {'dtype': torch.long, 'device':device}, [], [])) # 3
    # extended_attention_mask
    func_list.append(("bert_extended_attention_mask", [2, 1], {}, {'device': device}, [], [])) # 4
    # encoder_extended_attention_mask
    func_list.append(("bert_encoder_extended_attention_mask", [], {}, {}, [], [])) # 5
    # head mask: default as [None] * config.num_
    func_list.append(("bert_get_head_mask", [], {}, {}, [], [])) # 6

def _bert_embeddings(func_list, layers, device):
    """"""
    # position_ids
    func_list.append(("bert_position_ids", [1], {}, {'device': device}, [], [])) # 7
    # inputs_embeds
    _param, _hyper_param = extract_hyperparameters(layers[0][0], "embedding")
    func_list.append(("embedding", [0], _param, _hyper_param, [], [])) # 8
    # position_embeddings
    _param, _hyper_param = extract_hyperparameters(layers[2][0], "embedding")
    func_list.append(("embedding", [7], _param, _hyper_param, [], [])) # 9
    # token_type_embeddings
    _param, _hyper_param = extract_hyperparameters(layers[1][0], "embedding")
    func_list.append(("embedding", [3], _param, _hyper_param, [], [])) # 10
    # added up inputs_embeds and position_embedings
    func_list.append(("add", [8, 9], {}, {}, [], [])) # 11
    # added last embedding with token_type_embeddings
    func_list.append(("add", [11, 10], {}, {}, [], [])) # 12
    # layer_norm
    _param, _hyper_param = extract_hyperparameters(layers[3][0], "layer_norm")
    func_list.append(("layer_norm", [12], _param, _hyper_param, [], [])) # 13
    # dropout
    func_list.append(("dropout", [13], *extract_hyperparameters(layers[4][0], "dropout"), [], [])) # 14
    # output the embedding here, as the first hidden states


def _bert_layer(func_list, layers, i):
    """ bert layer starts layers[5]
    each one takes 10 sub-layers
    """
    base_idx = 5 + i * 12
    # --- attention layer
    # * self attention {query, key, value, dropout}
    func_list.append(("linear", [-1], *extract_hyperparameters(layers[base_idx][0], "linear"), [], [])) # mixed_query_layer # 15
    func_list.append(("linear", [-2], *extract_hyperparameters(layers[base_idx+1][0], "linear"), [], [])) # mixed_key_layer # 16
    func_list.append(("linear", [-3], *extract_hyperparameters(layers[base_idx+2][0], "linear"), [], [])) # mixed_value_layer # 17
    func_list.append(("bert_self_attn_trans_for_scores", [-3], {}, {}, [], [])) # query_layer # 18
    func_list.append(("bert_self_attn_trans_for_scores", [-3], {}, {}, [], [])) # key_layer # 19
    func_list.append(("bert_self_attn_trans_for_scores", [-3], {}, {}, [], [])) # value_layer # 20
    func_list.append(("transpose", [-2], {}, {"dim0": -1, "dim1": -2}, [], [])) # key_layer.transpose(-1, -2) # 21
    func_list.append(("matmul", [-4, -1], {}, {}, [], [])) # attention_scores # 22
    func_list.append(("bert_div", [-1], {}, {'y': 8}, [], [])) # attention_scores / math.sqrt(64)
    func_list.append(("softmax", [-1], {}, {"dim": -1}, [], [])) # attention_probs 
    func_list.append(("dropout", [-1], *extract_hyperparameters(layers[base_idx+3][0], "dropout"), [], [])) # self.dropout(attention_probs)
    func_list.append(("matmul", [-1, -6], {}, {}, [], [])) # context_layer # 26
    func_list.append(("bert_attn_proc_context", [-1], {}, {}, [], [])) # outputs # 27
    # * output {dense, dropout, layer_norm}
    func_list.append(("bert_idx", [-1], {}, {'i': 0}, [], [])) # self_outputs[0] # 28
    func_list.append(("linear", [-1], *extract_hyperparameters(layers[base_idx+4][0], "linear"), [], [])) # 29
    func_list.append(("dropout", [-1], *extract_hyperparameters(layers[base_idx+5][0], "dropout"), [], [])) # 30
    func_list.append(("add", [-1, -17], {}, {}, [], [])) # hidden_states + self_outputs[0] # 31
    func_list.append(("layer_norm", [-1], *extract_hyperparameters(layers[base_idx+6][0], "layer_norm"), [], [])) # attention_output # 32
    # === end attention layer
    
    # --- intermediate layer
    func_list.append(("linear", [-1], *extract_hyperparameters(layers[base_idx+7][0], "linear"), [], [])) # self.dense(attention_output) # 33
    func_list.append(("gelu", [-1], {}, {}, [], [])) # intermediate_output # 34
    # === end intermediate
    
    # --- self output layer
    func_list.append(("linear", [-1], *extract_hyperparameters(layers[base_idx+9][0], "linear"), [], [])) # self.dense(hidden_states) # 35
    func_list.append(("dropout", [-1], *extract_hyperparameters(layers[base_idx+10][0], "dropout"), [], [])) # self.dropout(hidden_states) #36
    func_list.append(("add", [-1, -5], {}, {}, [], [])) # hidden_states + input_tensor #37
    func_list.append(("layer_norm", [-1], *extract_hyperparameters(layers[base_idx+11][0], "layer_norm"), [], [])) # self.LayerNorm(hidden_states + input_tensor) #38
    # === end self output layer
    

def _bert_encoder(func_list, layers):
    """ 12 BertLayer: depend on 'bert-base-uncased' config
    """
    # first pass 
    for i in range(12):
        _bert_layer(func_list, layers, i)

def _bert_pool(func_list, layers):
    """ start from 138 layer
    """
    # base_idx = 137
    func_list.append(("bert_get_col", [-1], {}, {'c': 0}, [], [])) # first_token_tensor
    func_list.append(("linear", [-1], *extract_hyperparameters(layers[-2][0], "linear"), [], [])) # pooled_output
    func_list.append(("tanh", [-1], *extract_hyperparameters(layers[-1][0], "tanh"), [], [])) # pooled_output
    
    
def _make_func_list(layers, device):
    """ list elements: (func_name, input_index, params, hyperparams, forward_pre_hooks, forward_hooks)
    params can be set to None; but need to extract hyperparams
    """
    func_list = []
        
    # bert preparation layers
    _bert_prep(func_list, layers, device)
    # bert embedding layer
    _bert_embeddings(func_list, layers, device)
    # bert encoder
    _bert_encoder(func_list, layers)
    # bert pooler
    _bert_pool(func_list, layers)
    # combine sequence output and pooled results
    # func_list.append(("bert_combine", [-4, -1], {}, {}, [], []))
    
    return func_list


def import_model_reimpl(train=False, device='cuda'):
    layers = import_layer_list(train)
    func_list = _make_func_list(layers, device)
    return func_list

def import_model_reimpl_with_batching(train=False, device='cuda', max_batch_size=8 * 4096000):
    func_list = import_model_reimpl(train, device)
    return func_list, generate_batch_basic(func_list, max_batch_size=max_batch_size)

def print_layer_list():
    layers = import_layer_list()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(output_dir, 'bert_layer_list.txt'), 'w') as f:
        for i in range(len(layers)):
            mod, name = layers[i]
            f.write(f"{name}: {mod}\n")
        print(f"Saved layer list to {os.path.join(output_dir, 'bert_layer_list.txt')}")