import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
import os
import matplotlib.pyplot as plt


class ToTensor(object):
    """ convert ndarrays to tensor"""
    def __call__(delf, sample):
        image, label = sample['image'], sample['label']
        """
        numpy H W C
        torch C H W
        """
        image = image.transpose((2, 0, 1))
        sample['image'] = torch.as_tensor(image, dtype=torch.float)
        sample['label'] = torch.as_tensor(label, dtype=torch.long)
        return sample


class ADdataset(Dataset):
    def __init__(self, path, cv, transform=None):
        """
        Args:
            path (string): path to dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path = path
        self.transform = transform
        if (cv is None):
            self.files = [f for f in os.listdir(
                self.path) if os.isfile(os.path.join(self.path, f))]
        else:
            self.files = np.load(cv).tolist()

    def __len__(self):
        #files=[f for f in os.listdir(self.path) if os.isfile(os.path.join(self.path,f))]
        return len(self.files)

    def __getitem__(self, idx):
        # sample_name=os.path.join(self.root_dir,str(idx)+'.npy')
        sample_name = os.path.join(self.path, self.files[idx-1])
        sample = np.load(sample_name)
        image = sample[:, :, 0:3]
        label = sample[:, :, 3]

        # just in case in some patches, biomarkers are not marked as 1
        positive = label != 0
        negative = label == 0
        label[negative] = 0
        label[positive] = 1

        Sample = {'image': image, "label": label}

        if self.transform:
            Sample = self.transform(Sample)
        return Sample
