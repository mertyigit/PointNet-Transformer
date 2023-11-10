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
import torchvision
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

import open3d as o3
import math
import yaml


np.random.seed(0)
torch.manual_seed(0)

from tqdm import tqdm, trange

import matplotlib as mpl
import matplotlib.pyplot as plt


def TensorToImageGrid(images_batch, rows, cols):
    grid = torchvision.utils.make_grid(images_batch, nrow=cols)
    grid = grid.permute(1, 2, 0).cpu().numpy()
    plt.figure(figsize=(cols, rows))
    plt.imshow(grid, cmap='gray')
    plt.axis('off')
    return plt.show()

def TensorToImage(image):
    plt.imshow(image.numpy(), cmap='gray')
    return plt.show()