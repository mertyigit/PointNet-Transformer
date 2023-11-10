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
from .VisualTransformerEncoder import PositionWiseFeedForward

class VisualTransformerDecoderBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, d_ff, dropout):
        super(VisualTransformerDecoderBlock, self).__init__()
        self.self_attn = MultiHeadAttention(hidden_d, n_heads)
        self.cross_attn = MultiHeadAttention(hidden_d, n_heads)
        self.feed_forward = PositionWiseFeedForward(hidden_d, d_ff)
        self.norm1 = nn.LayerNorm(hidden_d)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.norm3 = nn.LayerNorm(hidden_d)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, enc_output, src_mask, tgt_mask):
        attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        return x

class VisualTransformerDecoder(nn.Module):
    def __init__(self, chw, d_ff, dropout, n_patches=7, num_layers=2, hidden_d=8, n_heads=2):
        # Super constructor
        super(VisualTransformerDecoder, self).__init__()

        # Attributes
        self.chw = chw  # ( C , H , W )
        self.n_patches = n_patches
        self.num_layers = num_layers
        self.n_heads = n_heads
        self.hidden_d = hidden_d
        self.d_ff = d_ff
        self.dropout = dropout
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
            [VisualTransformerDecoderBlock(hidden_d, n_heads, d_ff, dropout) for _ in range(num_layers)]
        )

        self.linear_decoder = nn.Linear(self.hidden_d, self.input_d)

    def forward(self, images, patches,  enc_output, src_mask, tgt_mask):
        # Image to Patches
        n, c, h, w = images.shape
        #patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        # Patch vector to Hidden Dimensions
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        #tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out, enc_output, src_mask, tgt_mask)
        
        out = self.linear_decoder(out)

        images = depatchify(out, self.n_patches, self.chw).to(self.positional_embeddings.device)

        return images