import sys


def extract_conv2d(layer):
    return (
        {
            'weight': layer.weight
        },
        {
            'stride': layer.stride,
            'padding': layer.padding,
            'dilation': layer.dilation,
            'groups': layer.groups
        }
    )


def extract_relu(layer):
    return (
        {},
        {
            'inplace': layer.inplace,
        }
    )


def extract_max_pool2d(layer):
    return (
        {},
        {
            'kernel_size': layer.kernel_size,
            'stride': layer.stride,
            'padding': layer.padding,
            'dilation': layer.dilation,
            'ceil_mode': layer.ceil_mode,
            'return_indices': layer.return_indices,
        }
    )


def extract_adaptive_avg_pool2d(layer):
    return (
        {},
        {
            'output_size': layer.output_size,
        }
    )


def extract_dropout(layer):
    return (
        {},
        {
            'p': layer.p,
            'training': layer.training,
            'inplace': layer.inplace
        }
    )


def extract_linear(layer):
    return (
        {
            'weight': layer.weight,
            'bias': layer.bias
        },
        {}
    )


def extract_batch_norm(layer):
    return (
        {
            'running_mean': layer.running_mean,
            'running_var': layer.running_var,
            'weight': layer.weight,
            'bias': layer.bias
        },
        {
            'training': layer.training,
            'momentum': layer.momentum,
            'eps': layer.eps
        }
    )


def extract_embedding(layer):
    params = {
        "weight": layer.weight
    }

    hyper_params = {
        'padding_idx': layer.padding_idx,
        'max_norm': layer.max_norm,
        'norm_type': layer.norm_type,
        'scale_grad_by_freq': layer.scale_grad_by_freq,
        'sparse': layer.sparse
    }
    return (params, hyper_params)


def extract_layer_norm(layer):
    params = {
        'weight': layer.weight,
        'bias': layer.bias
    }

    hyper_params = {
        'normalized_shape': layer.normalized_shape,
        'eps': layer.eps
    }

    return (params, hyper_params)


def extract_tanh(layer):
    params = {}

    hyper_params = {}

    return (params, hyper_params)


def extract_gpt2_conv1d(layer):
    params = {
        "weight": layer.weight,
        "bias": layer.bias
    }

    hyper_params = {
        "nf": layer.nf
    }
    return (params, hyper_params)


def extract_hyperparameters(layer, layer_name):
    if layer_name == 'conv2d':
        return extract_conv2d(layer)
    elif layer_name == 'relu':
        return extract_relu(layer)
    elif layer_name == 'max_pool2d':
        return extract_max_pool2d(layer)
    elif layer_name == 'adaptive_avg_pool2d':
        return extract_adaptive_avg_pool2d(layer)
    elif layer_name == 'dropout':
        return extract_dropout(layer)
    elif layer_name == 'linear':
        return extract_linear(layer)
    elif layer_name == 'batch_norm':
        return extract_batch_norm(layer)
    elif layer_name == "embedding":
        return extract_embedding(layer)
    elif layer_name == "layer_norm":
        return extract_layer_norm(layer)
    elif layer_name == "tanh":
        return extract_tanh(layer)
    elif layer_name == "gpt2_conv1d":
        return extract_gpt2_conv1d(layer)
    else:
        print('Extract Error: Undefiend layer', layer_name)
        sys.exit(0)
