import torch

def layer_init(layer, std=0.1, normal=False, gain=1.0):
    xavier = False if normal == True else True
    if normal:
        torch.nn.init.normal_(layer.weight, mean=0, std=std)
        torch.nn.init.constant_(layer.bias, 0)
        return
    elif xavier:
        torch.nn.init.xavier_normal_(layer.weight, gain=gain)
        torch.nn.init.constant_(layer.bias, 0)
        return
