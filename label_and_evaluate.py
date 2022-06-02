"""
This module contains utility functions for generating labels and optimal pair assignments for GradCAM outputs, as well as evaluating assignments
"""

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


def get_argmax_label(cam0, cam1, region_size=10):
    """Generates argmax labels from class activation maps from two classes
    
    Parameters
    ----------
    cam0 : torch.tensor
        Gradient weighed class activations for class 0
    cam1 : torch.tensor
        Gradient weighed class activations for class 1
    region_size : int
        Region size for morphological opening operation
    
    Return
    ------
    torch.tensor
        argmax labels
    """
    x00 = torch.arange(cam0.shape[0])*2+4.5
    x11 = torch.arange(cam0.shape[1])*2+4.5
    xorig = np.arange(132)
    Xorig = np.stack(np.meshgrid(xorig,xorig,indexing='ij'),-1)
    
    #interpolation
    values0 = np.asarray(cam0)
    values1 = np.asarray(cam1)
    resampledgradcam0 = interpn((x00, x11), values0, Xorig, bounds_error=False, fill_value=-1)
    resampledgradcam1 = interpn((x00, x11), values1, Xorig, bounds_error=False, fill_value=-1)

    #generated noisy argmax label
    argmax_label = torch.stack((torch.tensor(resampledgradcam1), torch.tensor(resampledgradcam0)))
    argmax_label = torch.argmax(argmax_label, dim=0)
    argmax_label = torch.tensor(argmax_label, dtype=torch.long)
    
    #replace edges with -1
    resampledgradcam0 = torch.tensor(resampledgradcam0, dtype=torch.long)
    argmax_label = torch.where(resampledgradcam0==-1, resampledgradcam0, torch.tensor(argmax_label, dtype=torch.long))
    
    #denoise argmax label
    denoised_label = opening(argmax_label, square(region_size))
    
    return denoised_label

def numbered_labels(argmax_label):
    """Generates numbered labels with Watershed segmentation
    
    Parameters
    ----------
    argmax_label : torch.tensor
        argmax labels
        
    Return
    ------
    torch.tensor
        Watershed segmented argmax labels
    """
    image = argmax_label
    distance = ndi.distance_transform_edt(image)
    coords = peak_local_max(distance, footprint=np.ones((7, 7)), labels=image)
    mask = np.zeros(distance.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)
    labels = watershed(-distance, markers, mask=image)
    
    return labels


def cost_matrix(A, B):
    """Generates cost matrix for a pair of labels based on centroid distance between regions
    
    Parameters
    ----------
    A : array
        Centroids of label set A
    B : array
        Centroids of label set B
        
    Return
    ------
    array
        Matrix of centroid distances
    """
    
    cost = np.zeros((len(A), len(B)))
    for i in range(len(A)):
        for j in range(len(B)):
            cost[i][j] = distance.euclidean(A[i], B[j])
    return cost

def evaluate_assignment(manual_label, generated_label, row_ind, col_ind):
    """Blob based evalation for generated labels
    
    Parameters
    ----------
    manual_label : torch.tensor
        numbered manual labeles
    generated_label : torch.tensor
        numbered generated labeles
    row_ind : list
        list of optimal assignments for generated label
    col_ind : list
        list of optimal assignments for manual label
        
    Return
    ------
    float
        true positive rate
    float
        false positive rate
    float
        false negative rate
    """
        
    TP = 0
    FP = len(np.unique(generated_label))-len(row_ind)-1 if len(row_ind) < len(np.unique(generated_label))-1 else 0
    FN = len(np.unique(generated_label))-len(row_ind)-1 if len(row_ind) > len(np.unique(generated_label))-1 else 0

    for i in range(len(row_ind)):
        argmax_idx = col_ind[i]
        manual_idx = row_ind[i]
        temp_TP = np.sum(generated_label[manual_label==manual_idx+1]==argmax_idx+1)
        if temp_TP > 0:
            TP += 1
        else:
            FP += 1
        print("manual label:", manual_idx, ", area:", np.count_nonzero(manual_label==manual_idx+1), ", argmax label:", argmax_idx, ", area:", np.count_nonzero(generated_label==argmax_idx+1), " Overlap:", temp_TP)
    
    precision = TP/(TP+FP) if TP+FP > 0 else 0
    recall = TP/(TP+FN) if TP+FN > 0 else 0
    
    print("precision:", precision, "recall:", recall)
    return TP, FP, FN

def evaluate(t, a):
    '''Pixel based evalation for generated labels
    
    Parameters
    ----------
    t : tensor
        manual labels
    a : tensor
        generated label

    '''
    
    t = np.array(t)
    a = np.array(a)
    TP = np.sum(t[a==1]==1)
    TN = np.sum(t[a==0]==0)
    FP = np.sum(t[a==0]==1)
    FN = np.sum(t[a==1]==0)
    
    #same as recall
    TPR = TP/(TP+FN) if TP+FN > 0 else 0
    TNR = TN/(TN+FP) if TN+FP > 0 else 0
    precision = TP/(TP+FP) if TP+FP > 0 else 0
    accuracy = (TP+TN)/(TP+TN+FP+FN) if TP+TN+FP+FN > 0 else 0
    
    return TPR, TNR, precision, accuracy