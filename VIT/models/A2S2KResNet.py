#!/usr/bin/env python
# coding: utf-8

# # Imports


import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.conv = nn.Conv2d(
            1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w, t = x.size()

        # feature descriptor on the global spatial information
        # 24, 1, 1, 1
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -3)).transpose(
            -1, -3).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)


class Residual(nn.Module):  # pytorch
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding,
            use_1x1conv=False,
            stride=1,
            start_block=False,
            end_block=False,
    ):
        super(Residual, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride), nn.ReLU())
        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride)
        if use_1x1conv:
            self.conv3 = nn.Conv3d(
                in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None

        if not start_block:
            self.bn0 = nn.BatchNorm3d(in_channels)

        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if start_block:
            self.bn2 = nn.BatchNorm3d(out_channels)

        if end_block:
            self.bn2 = nn.BatchNorm3d(out_channels)

        # ECA Attention Layer
        self.ecalayer = eca_layer(out_channels)

        # start and end block initialization
        self.start_block = start_block
        self.end_block = end_block

    def forward(self, X):
        identity = X

        if self.start_block:
            out = self.conv1(X)
        else:
            out = self.bn0(X)
            out = F.relu(out)
            out = self.conv1(out)

        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)

        if self.start_block:
            out = self.bn2(out)

        out = self.ecalayer(out)

        out += identity

        if self.end_block:
            out = self.bn2(out)
            out = F.relu(out)

        return out


class S3KAIResNet(nn.Module):
    def __init__(self, band, classes, reduction,PARAM_KERNEL_SIZE):
        super(S3KAIResNet, self).__init__()
        self.name = 'SSRN'
        self.conv1x1 = nn.Conv3d(
            in_channels=1,
            out_channels=PARAM_KERNEL_SIZE,
            kernel_size=(1, 1, 7),
            stride=(1, 1, 2),
            padding=0)
        self.conv3x3 = nn.Conv3d(
            in_channels=1,
            out_channels=PARAM_KERNEL_SIZE,
            kernel_size=(3, 3, 7),
            stride=(1, 1, 2),
            padding=(1, 1, 0))

        self.batch_norm1x1 = nn.Sequential(
            nn.BatchNorm3d(
                PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))
        self.batch_norm3x3 = nn.Sequential(
            nn.BatchNorm3d(
                PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.conv_se = nn.Sequential(
            nn.Conv3d(
                PARAM_KERNEL_SIZE, band // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True))
        self.conv_ex = nn.Conv3d(
            band // reduction, PARAM_KERNEL_SIZE, 1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.res_net1 = Residual(
            PARAM_KERNEL_SIZE,
            PARAM_KERNEL_SIZE, (1, 1, 7), (0, 0, 3),
            start_block=True)
        self.res_net2 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE,
                                 (1, 1, 7), (0, 0, 3))
        self.res_net3 = Residual(PARAM_KERNEL_SIZE, PARAM_KERNEL_SIZE,
                                 (3, 3, 1), (1, 1, 0))
        self.res_net4 = Residual(
            PARAM_KERNEL_SIZE,
            PARAM_KERNEL_SIZE, (3, 3, 1), (1, 1, 0),
            end_block=True)

        kernel_3d = math.ceil((band - 6) / 2)
        # print(kernel_3d)

        self.conv2 = nn.Conv3d(
            in_channels=PARAM_KERNEL_SIZE,
            out_channels=128,
            padding=(0, 0, 0),
            kernel_size=(1, 1, kernel_3d),
            stride=(1, 1, 1))
        self.batch_norm2 = nn.Sequential(
            nn.BatchNorm3d(128, eps=0.001, momentum=0.1, affine=True),  # 0.1
            nn.ReLU(inplace=True))
        self.conv3 = nn.Conv3d(
            in_channels=1,
            out_channels=PARAM_KERNEL_SIZE,
            padding=(0, 0, 0),
            kernel_size=(3, 3, 128),
            stride=(1, 1, 1))
        self.batch_norm3 = nn.Sequential(
            nn.BatchNorm3d(
                PARAM_KERNEL_SIZE, eps=0.001, momentum=0.1,
                affine=True),  # 0.1
            nn.ReLU(inplace=True))

        self.avg_pooling = nn.AvgPool3d(kernel_size=(5, 5, 1))
        self.full_connection = nn.Sequential(
            nn.Linear(PARAM_KERNEL_SIZE, classes)
            # nn.Softmax()
        )

    def forward(self, X):
        x_1x1 = self.conv1x1(X)
        x_1x1 = self.batch_norm1x1(x_1x1).unsqueeze(dim=1)
        x_3x3 = self.conv3x3(X)
        x_3x3 = self.batch_norm3x3(x_3x3).unsqueeze(dim=1)

        x1 = torch.cat([x_3x3, x_1x1], dim=1)
        U = torch.sum(x1, dim=1)
        S = self.pool(U)
        Z = self.conv_se(S)
        attention_vector = torch.cat(
            [
                self.conv_ex(Z).unsqueeze(dim=1),
                self.conv_ex(Z).unsqueeze(dim=1)
            ],
            dim=1)
        attention_vector = self.softmax(attention_vector)
        V = (x1 * attention_vector).sum(dim=1)

        x2 = self.res_net1(V)
        x2 = self.res_net2(x2)
        x2 = self.batch_norm2(self.conv2(x2))
        x2 = x2.permute(0, 4, 2, 3, 1)
        x2 = self.batch_norm3(self.conv3(x2))

        x3 = self.res_net3(x2)
        x3 = self.res_net4(x3)
        x4 = self.avg_pooling(x3)
        x4 = x4.view(x4.size(0), -1)
        self.x4 = x4
        return self.full_connection(x4)
    
    def extract_parameter(self):
        return self.x4


