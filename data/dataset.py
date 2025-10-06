from PIL import Image
import os
from torch.utils.data import Dataset
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root_z, root_h, transform=None):
        self.root_z = root_z
        self.root_h = root_h
        self.transform = transform

        self.z_images = os.listdir(self.root_z)
        self.h_images = os.listdir(self.root_h)
        self.len_dataset = max(len(self.z_images), len(self.h_images))
        self.z_len = len(self.z_images)
        self.h_len = len(self.h_images)

    def __len__(self):
        return self.len_dataset

    def __getitem__(self, idx):
        z_img = self.z_images[idx % self.z_len]
        h_img = self.h_images[idx% self.h_len]

        z_path = os.path.join(self.root_z, z_img)
        h_path = os.path.join(self.root_h, h_img)

        z_img = np.array(Image.open(z_path).convert('RGB'))
        h_img = np.array(Image.open(h_path).convert('RGB'))

        if self.transform:
            augmentations = self.transform(image=z_img, image0=h_img)
            z_img = augmentations['image']
            h_img = augmentations['image0']

        return z_img, h_img
