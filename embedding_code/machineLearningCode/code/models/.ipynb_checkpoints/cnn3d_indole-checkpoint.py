"""
3D CNN for 2x30x30x30 (2 channel volume)

Note: cannot do much 

12/07/2023, BC

"""
import os
import torch
import torch.nn as nn
from torchinfo import summary

# DDP
from torch.nn.parallel import DistributedDataParallel  as DDP
from torch.distributed import init_process_group, destroy_process_group


import logging
logger = logging.getLogger("GPU0")


class cnn3d(nn.Module):

    def __init__(self, in_channels=2, n_classes=3):
        super(cnn3d, self).__init__()
        self.conv1 = self._conv_relu_set(in_channels, 8)
        self.conv1p = self._conv_relu_pooling_set(8,16)
        self.conv2 = self._conv_relu_set(16, 16)
        self.conv2p = self._conv_relu_pooling_set(16, 32)
        self.conv3 = self._conv_relu_set(32, 32)
        self.conv3p = self._conv_relu_pooling_set(32,64)
        self.fc1 = nn.Linear(2*2*2*64, 128)
        self.fc2 = nn.Linear(128, n_classes)
        self.relu = nn.LeakyReLU()
        self.conv1_bn = nn.BatchNorm3d(16)
        self.conv2_bn = nn.BatchNorm3d(32)
        self.conv3_bn = nn.BatchNorm3d(64)
        self.fc1_bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(p=0.3)
        logger.info("         --> Use model: cnn3d.")

    def _conv_relu_set(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=(3, 3, 3), 
                stride=1,
                padding=1,
                ),
            nn.LeakyReLU(),
            )
        return conv_layer

    def _conv_relu_pooling_set(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=(3, 3, 3), 
                stride=1,
                padding=0,
                ),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            )
        return conv_layer

    def forward(self, x):
        #print('input shape:', x.shape)
        x = self.conv1(x)
        x = self.conv1p(x)
        x = self.conv1_bn(x)
        x = self.conv2(x)
        x = self.conv1_bn(x)
        x = self.conv2p(x)
        x = self.conv2_bn(x)
        x = self.conv3(x)
        x = self.conv2_bn(x)
        x = self.conv3p(x)
        x = self.conv3_bn(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc1_bn(x)
        x = self.drop(x)
        x = self.fc2(x)
        #print('output shape:', x.shape)

        return x



class cnn3d_2(nn.Module):

    def __init__(self, in_channels=2, n_classes=3):
        super(cnn3d_2, self).__init__()
        self.conv1 = self._conv_relu_set(in_channels, 16)
        self.conv1p = self._conv_relu_pooling_set(16,32)
        self.conv2 = self._conv_relu_set(32, 32)
        self.conv2p = self._conv_relu_pooling_set(32, 64)
        # self.conv3 = self._conv_relu_set(64, 64)
        # self.conv3p = self._conv_relu_pooling_set(64,128)
        # self.fc1 = nn.Linear(6*6*6*64, 256)  # for (30,30,30) input
        self.fc1 = nn.Linear(7*7*7*64, 256)  # for (35,35,35) input
        self.fc2 = nn.Linear(256, n_classes)
        self.relu = nn.LeakyReLU()
        self.conv1_bn = nn.BatchNorm3d(16)
        self.conv2_bn = nn.BatchNorm3d(32)
        self.conv3_bn = nn.BatchNorm3d(64)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.drop = nn.Dropout(p=0.3)
        logger.info("         --> Use model: cnn3d_2.")

    def _conv_relu_set(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=(3, 3, 3), 
                stride=1,
                padding=1,
                ),
            nn.LeakyReLU(),
            )
        return conv_layer

    def _conv_relu_pooling_set(self, in_channels, out_channels):
        conv_layer = nn.Sequential(
            nn.Conv3d(
                in_channels, 
                out_channels, 
                kernel_size=(3, 3, 3), 
                stride=1,
                padding=0,
                ),
            nn.LeakyReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2),
            )
        return conv_layer

    def forward(self, x):
        #print('input shape:', x.shape)
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.conv1p(x)
        x = self.conv2_bn(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.conv2p(x)
        x = self.conv3_bn(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc1_bn(x)
        x = self.drop(x)
        x = self.fc2(x)
        #print('output shape:', x.shape)

        return x

if __name__ == "__main__":
    model = cnn3d_2(in_channels=2, n_classes=3)
    summary(model, input_size=(5, 2, 35,35,35))