import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision.datasets.mnist import MNIST
from torchvision.transforms import ToTensor

import zipfile

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.loader import DataLoader
from torch_geometric.nn import MLP, fps, global_max_pool, radius
from torch_geometric.nn.conv import PointConv


# Custom Data Class for future applications
class MyDataset(Dataset):
    def __init__(self):
        pass
    
    
    def __len__(self):
        pass
    
    def __getitem__(self, idx):
        pass


class DataMNIST():
    """
    PyTorch Lightning data module 

    Args:
        data_dir:
        train_batch_size: 
        val_batch_size: 
        patch_size: 
        num_workers:
        pin_memory:
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        **kwargs,
    ):
        super().__init__()

        self.data_path = data_path
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = MNIST(
                            root=self.data_path,
                            train=True,
                            download = True,
                            transform=ToTensor()
        )
        
        self.val_dataset = MNIST(
                            root=self.data_path,
                            train=False,
                            download = True,
                            transform=ToTensor()
        )
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
                    self.train_dataset,
                    batch_size=self.train_batch_size,
                    shuffle=True,
                    pin_memory=False,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
                    self.val_dataset,
                    batch_size=self.val_batch_size,
                    shuffle=False,
                    pin_memory=False,
        )
    
