import importlib
import collections

model_map = collections.OrderedDict(sorted([
    ('resnet152', 'model.resnet.resnet152'),
    ('inception_v3', 'model.inception_v3.inception_v3'),
    ('bert_base', 'model.bert.bert_base'),
    ('densenet201', 'model.densenet.densenet201'),
    ('gpt2', 'model.gpt2.gpt2')
]))

def get_model_module(model_name):
    module_path = None
    for key in model_map:
        if model_name.startswith(key):
            module_path = model_map[key]
    if module_path is None:
        return None
    return importlib.import_module(module_path)