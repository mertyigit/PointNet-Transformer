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


from tqdm import tqdm, trange


def patchify(images, n_patches):
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches**2, h * w * c // n_patches**2, device='mps')
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches

def depatchify(patches, n_patches, chw):
    # size of patches is expected to be n, h, w
    patch_h = chw[1] // n_patches
    patch_w = chw[2] // n_patches
    n = patches.shape[0]

    images_recovered = torch.zeros(n, chw[0], chw[1], chw[2],  device='mps')

    for idx, patch in enumerate(patches):
        
        image_r = torch.empty(0, chw[2],  device='mps')
        
        for i in range(n_patches):
            #patch_r_i_1 = patch[i, :].view(patch_h, patch_w)
            row_tensor = torch.empty(patch_h, 0,  device='mps')
            for j in range(n_patches):
                patch_r_row = patch[i*n_patches+j, :].view(patch_h, patch_w)
                row_tensor = torch.cat((row_tensor, patch_r_row), dim=1)
            
            image_r = torch.cat((image_r, row_tensor), dim=0)

        images_recovered[idx] = image_r
    return images_recovered

def get_positional_embeddings(sequence_length, d):
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result