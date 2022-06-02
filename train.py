# -*- coding: utf-8 -*-
"""
Modified from UNET training code

Created on Sun Feb  2 16:39:39 2020
@author: Jasmine
This script is used to train a unet model from scratch or based on a pretrained model
* Train from scratch , it takes 4 arguments:         
        1) Trainset_path: path to training dataset
        2) batch_size: training batch size, recommend 16
        3) nepochs: number of epochs, recommend 3 as the dataset is quite big
        4) model_save_path: path to save the trained model
* Train based on pretrained model. it takes 5 arguments:    
        1) Trainset_path: path to training dataset
        2) batch_size: training batch size, recommend 16
        3) nepochs: number of epochs, recommend 3 as the dataset is quite big
        4) model_save_path: path to save the trained model
        5) model_load_path: path to pretrained model if you don't want to train from scratch
The up-to-date model will be saved
"""

import os
import sys
import numpy as np
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from network import CNN
# import data_augmentation
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'


class ToTensor(object) :
    """ convert ndarrays to tensor"""
    def __call__(delf,sample) :
        image,label=sample['image'],sample['label']
        """
        numpy H W C
        torch C H W
        """
        image=image.transpose((2,0,1))
        sample['image']=torch.as_tensor(image,dtype=torch.float)
        sample['label']=torch.as_tensor(label,dtype=torch.long)
        return sample
    
class ADdataset(Dataset):
    def __init__(self,path,cv,transform=None, random_orientation=True, gamma_transform=False, avg_noise=False, total_noise=False, blur=False):
        """
        Args:
            path (string): path to dataset
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.path = path
        self.transform = transform
        self.random_orientation = random_orientation
        self.gamma_transform = gamma_transform
        self.avg_noise = avg_noise
        self.total_noise = total_noise
        self.blur = blur

        if (cv is None):
            print("CV is None")
            self.files=[f for f in os.listdir(path) if os.path.isfile(os.path.join(path,f))] # list of file names
        else:
            self.files=np.load(cv).tolist() # list of file names
            print("# file names: ", len(self.files))
            # only keep actual files
            self.files = [i for i in self.files if os.path.exists(os.path.join(path, i))]
            print("# real file names: ", len(self.files))
    def __len__(self):
        #files=[f for f in os.listdir(self.path) if os.path.isfile(os.path.join(self.path,f))]
        return len(self.files)
    
    def __getitem__(self,idx):  
        #sample_name=os.path.join(self.path,str(idx)+'.npy')
        #sample=np.load(sample_name)
        sample_name = os.path.join(self.path,self.files[idx-1])
        sample = np.load(sample_name) # assuming idx begins with 1
        image=sample[:,:,0:3]
        label=sample[:,:,3]
        
        # just in case in some patches, biomarkers are not marked as 1
        positive=label!=0
        negative=label==0
        label[negative]=0
        label[positive]=1

        Sample={'image':image,"label":label}

        if self.transform :
            Sample=self.transform(Sample)     
               
        # Sample = data_augmentation.augment_sample(Sample, self.random_orientation, self.gamma_transform, self.avg_noise, self.total_noise, self.blur)

        return Sample

#save intermediate models
def save_ckp(e, model_state, checkpoint_dir, checkpoint_name):
    f_path = f"{checkpoint_dir}503_checkpoint_{checkpoint_name}_{e}.pth"
    torch.save(model_state, f_path)

def validate_unet(unet, dataloader) :
    correct=0
    TP = 0
    TN = 0
    FN = 0
    FP = 0
    for sample in dataloader:
        I=sample['image']
        tau = sample['label'].any(dim=1).any(dim=1)
        J = torch.as_tensor([1 if i else 0 for i in tau])
        Jhat=unet(I)
        output_labels = torch.as_tensor([1 if i[0]<i[1] else 0 for i in Jhat])

        #accuracy
        correct += torch.eq(J, output_labels).sum()
        TP += torch.logical_and(J, output_labels).sum()
        TN += torch.logical_not(torch.logical_and(J, output_labels)).sum()
        FP += torch.logical_and(torch.logical_xor(J, output_labels), torch.logical_not(J)).sum()
        FN += torch.logical_and(torch.logical_xor(J, output_labels), J).sum()
    accuracy = correct / len(dataloader.dataset)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    print("TP:", TP, "FP:", FP, "TN:",TN, "FN", FN)
    return accuracy, TPR, TNR

def train_unet(unet,dataloader,validation_dataloader,optimizer,loss_fn,nepochs,checkpoint_dir, checkpoint_name, radius=0):
    for e in range(nepochs):
        unet.train()
        loss_sum=0
        num_iteration=0
        correct = 0
        TP = 0
        TN = 0
        FN = 0
        FP = 0
        for sample in dataloader:
            I=sample['image']
            tau = sample['label'].any(dim=1).any(dim=1)
            J = torch.as_tensor([1 if i else 0 for i in tau])
            #J = torch.as_tensor([[0., 1.] if i else [1., 0.] for i in tau])
            Jhat = unet(I)
            loss =loss_fn(Jhat,J)
            loss_sum+=loss
            num_iteration+=1
            output_labels = torch.as_tensor([1 if i[0]<i[1] else 0 for i in Jhat])
            # update
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()
            #update metric values
            correct += torch.eq(J, output_labels).sum()
            TP += torch.logical_and(J, output_labels).sum()
            TN += torch.logical_not(torch.logical_and(J, output_labels)).sum()
            FP += torch.logical_and(torch.logical_xor(J, output_labels), torch.logical_not(J)).sum()
            FN += torch.logical_and(torch.logical_xor(J, output_labels), J).sum()
        #compute metrics
        mean_loss=loss_sum/num_iteration
        accuracy = correct / len(dataloader.dataset)
        TPR = TP / (TP + FN)
        TNR = TN / (TN + FP)
        print("The {}th epoch, mean loss={}, accuracy={}, TPR={}, TNR={}".format(e, mean_loss, accuracy, TPR, TNR))    

        unet.eval()
        with torch.no_grad():
            #compute validation accuracy, true positive and negative rates
            validation_acc, validation_TPR, validation_TNR = validate_unet(unet, validation_dataloader)
            print("validation accuracy={}, TPR={}, TNR={}\n".format(validation_acc, validation_TPR, validation_TNR))
        unet.train()

        if e % 10 == 0:
            #savae intermediate models
            save_ckp(e, unet.state_dict(), checkpoint_dir, checkpoint_name)
    return unet 


def main(argv):
    """
    Argv:
        1) Trainset_path: path to training dataset
        2) batch_size: training batch size, recommend 16
        3) nepochs: number of epochs, recommend 3 as the dataset is quite big
        4) model_save_path: path to save the trained model
        5) (optional) model_load_path: path to pretrained model if you don't want to train from scratch
    """
    Trainset_path=argv[0]  # path to training dataset
    batch_size=int(argv[1])
    nepochs=int(argv[2])   # number of epochs
    model_save_path=argv[3] #path to save the trained model 
    unet = CNN(in_ch=3, # number of channels in input image, RGB=3
            out_ch=2, # number of channels in output image, classification here is forground or background=2
            first_ch=8, # how many features at the first layer
            nmin=9, # minimum image size, this will define how big our input images need to be (it is printed)
           )
    cv=argv[4] # is None if use whole training dataset path, otherwise is list of files
    validationset_cv=argv[5]
    if (cv == "None"):
        cv = None
    checkpoint_name = argv[6]
    if len(argv)==7:
        print("We are going to train from scratch")
    elif len(argv)==8:
        model_load_path=argv[7]
        print("We are going to train from the pretrained model: ",model_load_path)
        unet.load_state_dict(torch.load(model_load_path))
    else:
        print("ilegal input argument")
    
    loss_fn=torch.nn.CrossEntropyLoss()
    learning_rate=1e-4
    optimizer = torch.optim.Adam(unet.parameters(), lr=learning_rate)
    AD_DataSet=ADdataset(path=Trainset_path,cv=cv,transform=ToTensor())
    AD_DataLoader=DataLoader(AD_DataSet,batch_size=batch_size,shuffle=True,num_workers=0)

    validationset=ADdataset(path=Trainset_path,cv=validationset_cv,transform=ToTensor())
    validation_DataLoader=DataLoader(validationset,batch_size=1,shuffle=False,num_workers=0)
    print("The unet architecture:",unet)
    print("Training....")
    unet=train_unet(unet=unet,
                    dataloader=AD_DataLoader,
                    validation_dataloader=validation_DataLoader,
                    optimizer=optimizer,
                    loss_fn=loss_fn,
                    nepochs=nepochs,
                    checkpoint_dir="/Users/ophelia/Desktop/TrainingDataApril2022/models/",
                   checkpoint_name=checkpoint_name)
    torch.save(unet.state_dict(),model_save_path)
    print("Finished!")
    print("New model has been saved in: ", model_save_path)

    
if __name__ == '__main__':
  main(sys.argv[1:])