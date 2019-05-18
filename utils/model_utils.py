import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import numpy as np



def init_weights(m):
    if isinstance(m, nn.Linear):
        I.normal_(m.weight, std=0.14)

def rotation(xy_real, z_pred):
    #xy_real = torch.from_numpy(xy_real)
    #x_real, y_real = xy_real[:, :1], xy_real[:, 1:]
    pass


def heuristic_loss(xy_real, z_pred):
    pass




















