import time
import math
import os
import sys
import numpy as np

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F

import collections

# CIFAR10 32*32
class CIFAR10CNN(nn.Module):
    def __init__(self, NChannels):
        super(CIFAR10CNN, self).__init__()
        self.features = []

        #3*32*32
        self.conv11 = nn.Conv2d(
            in_channels = NChannels,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv11)

        self.Relu11 = nn.ReLU()
        self.features.append(self.Relu11)

        #64*32*32
        self.conv12 = nn.Conv2d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv12)

        self.Relu12 = nn.ReLU()
        self.features.append(self.Relu12)

        # 64 * 32 * 32
        self.pool1 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool1)

        #64 * 16 * 16
        self.conv21 = nn.Conv2d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv21)

        self.Relu21 = nn.ReLU()
        self.features.append(self.Relu21)

        #128 * 16 * 16
        self.conv22 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv22)

        self.Relu22 = nn.ReLU()
        self.features.append(self.Relu22)

        # 128 * 16 * 16
        self.pool2 = nn.MaxPool2d(2, 2)
        self.features.append(self.pool2)

        #128 * 8 * 8
        self.conv31 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv31)

        #128 * 8 * 8
        self.Relu31 = nn.ReLU()
        self.features.append(self.Relu31)

        #128 * 8 * 8
        self.conv32 = nn.Conv2d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.features.append(self.conv32)

        self.Relu32 = nn.ReLU()
        self.features.append(self.Relu32)
        # 128 * 8 * 8
        self.pool3 = nn.MaxPool2d(2, 2)
        #128 * 4 * 4
        self.features.append(self.pool3)

        self.classifier = []
        self.feature_dims = 4 * 4 * 128
        self.fc1 = nn.Linear(self.feature_dims, 512)
        self.classifier.append(self.fc1)

        # self.fc1act = nn.Sigmoid()
        self.fc1act = nn.ReLU()
        self.classifier.append(self.fc1act)

        self.fc2 = nn.Linear(512, 10)
        self.classifier.append(self.fc2)

    def forward(self, x):

        for layer in self.features:
            x = layer(x)

        x = x.view(-1, self.feature_dims)

        for layer in self.classifier:
            x = layer(x)

        return x


























