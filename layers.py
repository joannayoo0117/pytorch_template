import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLayer(nn.Module):
    def __init__(self):
        super(CustomLayer, self).__init__()

    def forward(self, input):
        pass