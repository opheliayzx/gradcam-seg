import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim


class CNN(nn.Module):
    """
    CNN for weakly supervised learning
    input: RGB images
    output: probability of each class 
    """

    def DCC(self, in_ch, out_ch):
        block = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_ch, out_ch, 3),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3),
            nn.ReLU()
        )
        return block

    # Final downsample convolution without ReLU
    def final_DCC(self, in_ch, out_ch):
        block = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(in_ch, out_ch, 3),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3),
        )
        return block

    def __init__(self, in_ch, out_ch, first_ch, nmin=9):
        """
        in_ch: number of channels in input image, RGB=3
        out_ch: number of channels in output, classification here is no Tau or Tau (2)
        first_ch: number of features at the first layer
        """
        super(CNN, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.first_ch = first_ch
        self.nmin = nmin
        # First layer
        self.cinput1 = nn.Conv2d(in_ch, first_ch, 3)
        self.cinput2 = nn.Conv2d(first_ch, first_ch, 3)
        # Downsampling
        self.dcc1 = self.DCC(first_ch, first_ch*2)
        self.dcc2 = self.DCC(first_ch*2, first_ch*4)
        self.dcc3 = self.final_DCC(first_ch*4, first_ch*8)
        # reshaping the output to batch size * 2
        self.linear1 = nn.Linear(first_ch*8*81, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, 2)

    def forward(self, x):
        print("original", x)
        x = self.cinput1(x)
        print("input1", x.shape)
        x = self.cinput2(x)
        print("input2", x.shape)

        x = self.dcc1(x)
        print("dcc1", x.shape)
        x = self.dcc2(x)
        print("dcc2", x.shape)
        x = self.dcc3(x)

        x = x.view(x.size(dim=0), -1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x
