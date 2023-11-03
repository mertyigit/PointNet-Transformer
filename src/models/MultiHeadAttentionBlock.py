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


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_d, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert hidden_d % n_heads == 0, f"{hidden_d} cannot be divided into {n_heads} heads!"

        self.hidden_d = hidden_d
        self.n_heads = n_heads
        self.d_k = hidden_d // n_heads ## Hidded dimensions per head
        
        # Define the linear transformation layers for Query, Key, and Value
        self.W_q = nn.Linear(hidden_d, hidden_d)
        self.W_k = nn.Linear(hidden_d, hidden_d)
        self.W_v = nn.Linear(hidden_d, hidden_d)
        self.W_o = nn.Linear(hidden_d, hidden_d)
        
    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Apply the softmax function to calculated the scaled dot-product attention scores to get attention weights
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V) # Calculate the weighted sum using the attention weights
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, hidden_d = x.size()
        return x.view(batch_size, seq_length, self.n_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.hidden_d)
        
    def forward(self, Q, K, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output