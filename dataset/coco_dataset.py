import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import numpy as np
import json
import glob
import torch

class COCOImageDataset(Dataset):
    def __init__(self, folder, transform=None, target_transform=None):
        # Load labels and imgs
        dir_name = os.path.split(folder)[-1]
        self.folder = os.path.abspath(folder)
        self.img_dir = os.path.join(self.folder, f"{dir_name}_Images")
        self.img_labels = pd.read_csv(os.path.join(self.folder, f"{dir_name}_annotations.txt"), sep=",", header=None)
        self.img_labels = self.img_labels.drop(list(range(5, 13)), axis=1)
        self.img_labels = self.img_labels.rename(columns={0: "img_name", 1: "xmin", 2: "xmax", 3: "ymin", 4: "ymax", 13: "class"})
        self.images = self.img_labels['img_name'].unique().tolist()

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        image is C X H X W Tensor
        label is N X 4 Tensor where N is number of hands
        """
        img_name = self.images[idx]
        all_label = self.img_labels.loc[self.img_labels['img_name'] == img_name]
        img_path = os.path.join(self.img_dir, img_name)
        image = read_image(img_path)
        label = torch.from_numpy(all_label[['xmin', 'xmax', 'ymin', 'ymax']].to_numpy())
        
        return image, label