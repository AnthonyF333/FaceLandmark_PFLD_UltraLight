#!/usr/bin/env python3
# -*- coding:utf-8 -*-


import torch
from torch.nn import Module, AvgPool2d, Linear
from models.base_module import Conv_Block, GhostBottleneck
import torch.nn.functional as F


class PFLD_Ultralight(Module):
    def __init__(self, width_factor=1, input_size=112, landmark_number=98):
        super(PFLD_Ultralight, self).__init__()

        self.conv1 = Conv_Block(3, int(64 * width_factor), 3, 2, 1)
        self.conv2 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1, group=int(64 * width_factor))

        self.conv3_1 = GhostBottleneck(int(64 * width_factor), int(128 * width_factor), int(80 * width_factor), stride=2)
        self.conv3_2 = GhostBottleneck(int(80 * width_factor), int(160 * width_factor), int(80 * width_factor), stride=1)
        self.conv3_3 = GhostBottleneck(int(80 * width_factor), int(160 * width_factor), int(80 * width_factor), stride=1)

        self.conv4_1 = GhostBottleneck(int(80 * width_factor), int(240 * width_factor), int(96 * width_factor), stride=2)
        self.conv4_2 = GhostBottleneck(int(96 * width_factor), int(288 * width_factor), int(96 * width_factor), stride=1)
        self.conv4_3 = GhostBottleneck(int(96 * width_factor), int(288 * width_factor), int(96 * width_factor), stride=1)

        self.conv5_1 = GhostBottleneck(int(96 * width_factor), int(384 * width_factor), int(144 * width_factor), stride=2)
        self.conv5_2 = GhostBottleneck(int(144 * width_factor), int(576 * width_factor), int(144 * width_factor), stride=1)
        self.conv5_3 = GhostBottleneck(int(144 * width_factor), int(576 * width_factor), int(144 * width_factor), stride=1)
        self.conv5_4 = GhostBottleneck(int(144 * width_factor), int(576 * width_factor), int(144 * width_factor), stride=1)

        self.conv6 = GhostBottleneck(int(144 * width_factor), int(288 * width_factor), int(16 * width_factor), stride=1)
        self.conv7 = Conv_Block(int(16 * width_factor), int(32 * width_factor), 3, 1, 1)
        self.conv8 = Conv_Block(int(32 * width_factor), int(128 * width_factor), input_size // 16, 1, 0, has_bn=False)

        self.avg_pool1 = AvgPool2d(input_size // 2)
        self.avg_pool2 = AvgPool2d(input_size // 4)
        self.avg_pool3 = AvgPool2d(input_size // 8)
        self.avg_pool4 = AvgPool2d(input_size // 16)

        self.fc = Linear(int(512 * width_factor), landmark_number * 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x3 = self.avg_pool3(x)
        x3 = x3.view(x3.size(0), -1)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = self.conv5_4(x)
        x4 = self.avg_pool4(x)
        x4 = x4.view(x4.size(0), -1)

        x = self.conv6(x)
        x = self.conv7(x)
        x5 = self.conv8(x)
        x5 = x5.view(x5.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3, x4, x5], 1)
        landmarks = self.fc(multi_scale)

        return landmarks


class PFLD_Ultralight_AuxiliaryNet(Module):
    def __init__(self, width_factor=1):
        super(PFLD_Ghost_AuxiliaryNet, self).__init__()
        self.conv1 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 1, 1, 0)
        self.conv2 = Conv_Block(int(80 * width_factor), int(64 * width_factor), 1, 1, 0)
        self.conv3 = Conv_Block(int(96 * width_factor), int(64 * width_factor), 1, 1, 0)
        self.conv4 = Conv_Block(int(144 * width_factor), int(64 * width_factor), 1, 1, 0)

        self.merge1 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1)
        self.merge2 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1)
        self.merge3 = Conv_Block(int(64 * width_factor), int(64 * width_factor), 3, 1, 1)

        self.conv_out = Conv_Block(int(64 * width_factor), 1, 1, 1, 0)

    def forward(self, out1, out2, out3, out4):
        output1 = self.conv1(out1)
        output2 = self.conv2(out2)
        output3 = self.conv3(out3)
        output4 = self.conv4(out4)

        up4 = F.interpolate(output4, size=[output3.size(2), output3.size(3)], mode="nearest")
        output3 = output3 + up4
        output3 = self.merge3(output3)

        up3 = F.interpolate(output3, size=[output2.size(2), output2.size(3)], mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2, size=[output1.size(2), output1.size(3)], mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        output1 = self.conv_out(output1)

        return output1
