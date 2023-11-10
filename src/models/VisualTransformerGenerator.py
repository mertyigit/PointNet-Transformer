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

from .VisualTransformerEncoder import VisualTransformerEncoder
from .VisualTransformerDecoder import VisualTransformerDecoder

class Transformer(nn.Module):
    def __init__(self, hidden_d, n_heads, num_layers, d_ff, dropout, n_patches):
        super(Transformer, self).__init__()

        self.n_heads = n_heads
        self.hidden_d = hidden_d
        self.num_layers = num_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.n_patches = n_patches

        self.encoder = VisualTransformerEncoder((1, 28, 28), n_patches=self.n_patches, num_layers=self.num_layers, hidden_d=self.hidden_d, n_heads=self.n_heads)
        self.decoder = VisualTransformerDecoder((1, 28, 28), d_ff=self.d_ff, dropout=self.dropout, n_patches=self.n_patches, num_layers=self.num_layers, hidden_d=self.hidden_d, n_heads=self.n_heads)

        
    def generate_mask(self, patches):
        src = torch.ones((patches.size(0), patches.size(1)), device='mps')
        tgt = torch.ones((patches.size(0), patches.size(1)), device='mps')

        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)
        seq_length = tgt.size(1)
        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length, device='mps'), diagonal=1)).bool()
        tgt_mask = tgt_mask & nopeak_mask
        nopeak_mask = nopeak_mask.detach()
        del nopeak_mask
        return src_mask, tgt_mask

    def forward(self, images):
        enc_output, patches = self.encoder(images)
        src_mask, tgt_mask = self.generate_mask(patches)
        #src_mask = src_mask.to(device='mps')
        #tgt_mask = tgt_mask.to(device='mps')
        dec_output = self.decoder(images, patches, enc_output, src_mask, tgt_mask)

        src_mask = src_mask.detach()
        tgt_mask = tgt_mask.detach()
        del src_mask
        del tgt_mask
        
        return dec_output