import torch

def create_criterion(loss_name, reduction='sum'):
    if loss_name == "cross_entropy":
        return torch.nn.CrossEntropyLoss(reduction=reduction)
    else:
        return None