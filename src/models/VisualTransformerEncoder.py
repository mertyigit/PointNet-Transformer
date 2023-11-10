import os
import sys
import re
from glob import glob
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import MulticlassMatthewsCorrCoef
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.functional import kl_div
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

import open3d as o3
import math
import yaml
import argparse

from ..utils.features import *
from .MultiHeadAttentionBlock import *

class PositionWiseFeedForward(nn.Module):
    def __init__(self, hidden_d, d_ff):
        super(PositionWiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(hidden_d, d_ff)
        self.fc2 = nn.Linear(d_ff, hidden_d)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
        
class VisualTransformerEncoderBlock(nn.Module):
    '''
    Not typical transformer block, the normalization and linear layers are there but the order is different.
    '''
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(VisualTransformerEncoderBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MultiHeadAttention(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x), self.norm1(x), self.norm1(x)) 
        out = out + self.mlp(self.norm2(out))
        return out

    
class VisualTransformerEncoder(nn.Module):
    def __init__(self, chw, n_patches=7, num_layers=2, hidden_d=8, n_heads=2):
        # Super constructor
        super(VisualTransformerEncoder, self).__init__()

        # Attributes
        self.chw = chw  # ( C , H , W )
        self.n_patches = n_patches
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert (
            chw[1] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        assert (
            chw[2] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        #self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(n_patches**2, hidden_d),
            persistent=False,
        )

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [VisualTransformerEncoderBlock(hidden_d, n_heads) for _ in range(num_layers)]
        )

    def forward(self, images):
        # Image to Patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Patch vector to Hidden Dimensions
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        #tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)
        
        return out, patches