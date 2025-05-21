import torch
import math

def input_size(input, **kwargs):
    return input.size()


def bert_extended_attention_mask(attention_mask, input_shape, device, **kwargs):
    """ copied from huggingface """
    if attention_mask.dim() == 3:
        extended_attention_mask = attention_mask[:, None, :, :]
    elif attention_mask.dim() == 2:
        # assume not be a decoder
        extended_attention_mask = attention_mask[:, None, None, :]
    else:
        raise ValueError(
            "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                input_shape, attention_mask.shape
            )
        )
    # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
    extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
    return extended_attention_mask


def bert_encoder_extended_attention_mask(**kwargs):
    """ Assume the input does not contain `encoder_hidden_states` """
    return None

def bert_get_head_mask(**kwargs):
    """ this is 12 from configuration of 'bert-base-uncased' """
    return [None] * 12


def bert_position_ids(input_shape, device):
    seq_length = input_shape[1]
    # pylint: disable=no-member
    position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
    position_ids = position_ids.unsqueeze(0).expand(input_shape)
    return position_ids

def bert_self_attn_trans_for_scores(x):
    new_x_shape = x.size()[:-1] + (12, 64) # (12, 64) is fixed for 'bert-base-uncased'
    x = x.view(*new_x_shape)
    return x.permute(0, 2, 1, 3)


def bert_div(x, y):
    return x / y

def bert_idx(x, i):
    return x[i]

def bert_get_col(x, c):
    return x[:, c]

def bert_combine(*args):
    ret = []
    for a in args:
        ret.append(a)
    return ret

def bert_attn_proc_context(context_layer):
    """copy from huggingface code"""
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (768,) # fixed for 'bert-base-uncased'
    context_layer = context_layer.view(*new_context_layer_shape)

    # outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
    outputs = (context_layer,) # self.output_attentions is False
    return outputs

def list_append(l, e):
    l.append(e)
    return l

def inception_transform_input(x):
    # pylint: disable=no-member
    x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
    x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
    x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
    x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
    return x

def make_list(*args):
    ret = []
    for a in args:
        ret.append(a)
    return ret


def gpt2_conv1d(x, weight, bias, nf):
    """"""
    size_out = x.size()[:-1] + (nf,)
    x = torch.addmm(bias, x.view(-1, x.size(-1)), weight) # pylint: disable=no-member
    x = x.view(*size_out)
    return x

def gpt2_gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    # pylint: disable=no-member
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gpt2_attn_w(q, k, v, device):
    """ first part of self._attn function call """
    w = torch.matmul(q, k) # pylint:disable=no-member
    w = w / math.sqrt(v.size(-1))
    nd, ns = w.size(-2), w.size(-1)
    b = torch.tril(torch.ones(nd,ns, device=device).view(1, 1, nd, ns)) # pylint:disable=no-member
    w = w * b - 1e4 * (1 - b)
    return w


def gpt2_split_heads(inputs, idx, k=False):
    """ self.n_head = 12 """
    x = inputs[idx]
    # new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
    new_x_shape = x.size()[:-1] + (12, x.size(-1) // 12)
    x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
    if k:
        return x.permute(0, 2, 3, 1)  # (batch, head, head_features, seq_length)
    else:
        return x.permute(0, 2, 1, 3)  # (batch, head, seq_length, head_features)


def gpt2_merge_head(x):
    """"""
    x = x.permute(0, 2, 1, 3).contiguous()
    new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
    return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

def gpt2_output_shape(input_shape, hidden_states):
    output_shape = input_shape + (hidden_states.size(-1),)
    return output_shape

def get_ith(x, i):
    return x[i]

def tensor_view(t, *shape):
    return t.view(*shape)

def put_val(v):
    return v

def empty_list():
    return []

def tensor_shape(t):
    return t.shape

def gpt2_add(*args):
    ret = args[0]
    for i in range(1, len(args)):
        ret += args[i]
    return ret
