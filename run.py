'''
Train Validate Evaluation Script
'''

import os
import sys
import re
from glob import glob
import time
import numpy as np
import pandas as pd

### TORCH ###
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from torchmetrics.classification import MulticlassMatthewsCorrCoef
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn.functional import kl_div
import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, fps, global_max_pool, radius
from torch_geometric.nn.conv import PointConv
##############

import logging
import open3d as o3
import yaml
import argparse



### IN-HOUSE CODES ###
from src.models.PointNetEncoder import PointNetBackbone
from src.utils.calculate_loss import ChamferDistanceLoss
from src.models.PointCloudEncoder import PointCloudEncoder
from src.models.PointCloudDecoder import PointCloudDecoder, PointCloudDecoderSelf, PointCloudDecoderMLP
from src.models.AutoEncoder import AutoEncoder
from src.models.VAE import VAE
from src.data.dataset import DataModelNet
from src.utils.utils import *
from src.utils.train import Trainer
#######################

from tqdm import tqdm

import matplotlib as mpl
import matplotlib.pyplot as plt

import wandb

### HERE ARGS ###
parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

#################

### REPRODUCIBILITY ###
torch.seed = config['trainer_parameters']['manual_seed']
#######################

#### W&B INIT ###
#wandb.init(
#    # set the wandb project where this run will be logged
#    project="PointNet-VAE",
#    
#    # track hyperparameters and run metadata
#    config=config
#)
###############


print('MPS is build: {}'.format(torch.backends.mps.is_built()))
print('MPS Availability: {}'.format(torch.backends.mps.is_available()))
DEVICE = 'cuda' if torch.cuda.is_available() else 'mps'
print('Device is set to :{}'.format(DEVICE))


# model hyperparameters
EPOCHS = config['trainer_parameters']['epochs']
LR = config['trainer_parameters']['lr']
LATENT_DIM = config['model_parameters']['latent_dim']
DEVICE = config['model_parameters']['device']

#### LOAD DATA ####
data = DataModelNet(**config["data_parameters"])
data.setup()
train_dataloader = data.train_dataloader()
val_dataloader = data.val_dataloader()
###################

transformer = Transformer(hidden_d, n_heads, num_layers, d_ff, dropout, n_patches).to(DEVICE)

model_run = Trainer(model=vae, 
                    criterion=ChamferDistanceLoss(),
                    optimizer=optim.Adam(vae.parameters(), config['trainer_parameters']['lr']),
                    **config['model_parameters']
                    )

model_run.fit(train_dataloader, val_dataloader, EPOCHS)












