#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
from torch.nn import Module, AvgPool2d, Linear
from models.base_module import Conv_Block, InvertedResidual


class PFLD(Module):
    def __init__(self, width_factor=1, input_size=112, landmark_number=98):
        super(PFLD, self).__init__()

        self.conv1 = Conv_Block(3, int(64 * width_factor), 3, 2, 1)
        self.conv2 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1, group=int(64 * width_factor))

        self.conv3_1 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 2, False, 2)
        self.conv3_2 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 1, True, 2)
        self.conv3_3 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 1, True, 2)
        self.conv3_4 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 1, True, 2)
        self.conv3_5 = InvertedResidual(int(64 * width_factor), int(64 * width_factor), 1, True, 2)

        self.conv4 = InvertedResidual(int(64 * width_factor), int(128 * width_factor), 2, False, 2)

        self.conv5_1 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, False, 4)
        self.conv5_2 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)
        self.conv5_3 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)
        self.conv5_4 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)
        self.conv5_5 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)
        self.conv5_6 = InvertedResidual(int(128 * width_factor), int(128 * width_factor), 1, True, 4)

        self.conv6 = InvertedResidual(int(128 * width_factor), int(16 * width_factor), 1, False, 2)
        self.conv7 = Conv_Block(int(16 * width_factor), int(32 * width_factor), 3, 2, 1)
        self.conv8 = Conv_Block(int(32 * width_factor), int(128 * width_factor), input_size // 16, 1, 0, has_bn=False)

        self.avg_pool1 = AvgPool2d(input_size // 8)
        self.avg_pool2 = AvgPool2d(input_size // 16)
        self.fc = Linear(int(176 * width_factor), landmark_number * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)
        x = self.conv3_5(x)

        x = self.conv4(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x = self.conv5_5(x)
        x = self.conv5_6(x)

        x = self.conv6(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.conv8(x)
        x3 = x3.view(x1.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return landmarks
