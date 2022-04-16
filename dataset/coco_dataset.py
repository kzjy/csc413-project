import os
from torch.utils.data import Dataset
import cv2
import pandas as pd
import numpy as np
import json
import glob
import torch

class COCOImageDataset(Dataset):
    def __init__(self, folder, transform=None):
        # Load labels and imgs
        dir_name = os.path.split(folder)[-1]
        self.folder = os.path.abspath(folder)
        self.img_dir = os.path.join(self.folder, f"{dir_name}_Images")
        self.img_labels = pd.read_csv(os.path.join(self.folder, f"{dir_name}_annotations.txt"), sep=",", header=None)
        self.img_labels = self.img_labels.drop(list(range(5, 13)), axis=1)
        self.img_labels = self.img_labels.rename(columns={0: "img_name", 1: "xmin", 2: "xmax", 3: "ymin", 4: "ymax", 13: "class"})
        self.images = self.img_labels['img_name'].unique().tolist()

        self.transform = transform
        # self.target_transform = target_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        image is C X H X W Tensor
        target is dict
            label: N X 4 Tensor where N is number of hands
        """
        img_name = self.images[idx]
        all_label = self.img_labels.loc[self.img_labels['img_name'] == img_name]
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path, cv2.COLOR_BGR2RGB)
        bbox = all_label[['xmin','ymin', 'xmax','ymax']].to_numpy()
        zeros = np.zeros(bbox.shape[0]).reshape((bbox.shape[0], 1))
        # labels = all_label[['xmin','ymin', 'xmax','ymax']].values.tolist()


        bbox = np.hstack((bbox, zeros))

        data = {'img': image, 'annot': bbox}

        if self.transform is not None:
            data = self.transform(data)
        
        # image = torch.from_numpy(image).permute(2, 0, 1).float()
        # target = {}
        # target['bbox'] = torch.tensor(bbox)
        # target['class'] = torch.tensor([0])
        
        # print(image.size(), target['bbox'].size())
        # print(data)
        return data