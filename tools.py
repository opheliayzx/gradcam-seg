import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from scipy.interpolate import interpn
from skimage.morphology import closing, opening
from skimage.morphology import square, disk

from network import CNN
from load_data import ToTensor, ADdataset
from extractor import feature_extractor
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy import ndimage as ndi
from skimage.feature import peak_local_max

from skimage.segmentation import watershed
from skimage.segmentation import random_walker


def get_gradcam(act0, grad0, act1, grad1):
    cam0 = torch.sum(act0 * torch.mean(grad0[0],dim=(-1,-2),keepdim=True),dim=(0,1))
    cam1 = torch.sum(act1 * torch.mean(grad1[0],dim=(-1,-2),keepdim=True),dim=(0,1))
    
    return cam0, cam1

def plot_gradcam(act0, grad0, act1, grad1):
    fig,ax = plt.subplots(1, 2)
    cam0, cam1 = get_gradcam(act0, grad0, act1, grad1)
    
    #class 0 (No Tau)
    ax[0].imshow(cam0)
    ax[0].set_title('Class 0 (No Tau)')

    #class 1 (Tau)
    ax[1].imshow(cam1)
    ax[1].set_title('Class 1 (Tau)')
    
    return fig, ax, cam0, cam1

def plot_comparison(original, filtered, filter_name):

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original, cmap=plt.cm.gray)
    ax1.set_title('original')
    ax1.axis('off')
    ax2.imshow(filtered, cmap=plt.cm.gray)
    ax2.set_title(filter_name)
    ax2.axis('off')

    
    
def plot_difference(t, a):
    diff = torch.eq(t, a)
    f, a = plt.subplots()
    a.imshow(diff)
    
