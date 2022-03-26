import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import numpy as np
import json
import glob

class CMUImageDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        pass
        
    def __len__(self):
        return 0

    def __getitem__(self, idx):
        return None