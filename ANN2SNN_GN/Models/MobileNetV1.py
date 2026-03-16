'''
MobileNetV1 implementation in pytorch. Architecture is based on the Tensorflow implementation used in MLPerf Tiny: https://github.com/mlcommons/tiny/blob/master/benchmark/training/visual_wake_words/vww_model.py
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.depthwise = nn.Conv2d(in_channels,in_channels,kernel_size=3,stride=stride,padding=1,groups=in_channels,bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.bn1(self.depthwise(x)))   
        x = self.relu2(self.bn2(self.pointwise(x)))     
        return x

class MobileNetV1(nn.Module):

    def __init__(self, num_classes=2, num_filters=32, strideFistConv=1):
        super().__init__()

        num_filters = num_filters  # alpha = 0.25 (num_filters=8) version for coco. alpha = 1 (num_filters=32) for cifar10/100 because the network looses spatial information too fast

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, num_filters, kernel_size=3, stride=strideFistConv, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )
        self.ds2 = DepthwiseSeparableConv(num_filters, num_filters * 2, stride=1)
        num_filters *= 2

        self.ds3 = DepthwiseSeparableConv(num_filters, num_filters * 2, stride=2)
        num_filters *= 2

        self.ds4 = DepthwiseSeparableConv(num_filters, num_filters, stride=1)

        self.ds5 = DepthwiseSeparableConv(num_filters, num_filters * 2, stride=2)
        num_filters *= 2

        self.ds6 = DepthwiseSeparableConv(num_filters, num_filters, stride=1)

        self.ds7 = DepthwiseSeparableConv(num_filters, num_filters * 2, stride=2)
        num_filters *= 2

        # layers 8-12
        self.ds8 = DepthwiseSeparableConv(num_filters, num_filters, stride=1)
        self.ds9 = DepthwiseSeparableConv(num_filters, num_filters, stride=1)
        self.ds10 = DepthwiseSeparableConv(num_filters, num_filters, stride=1)
        self.ds11 = DepthwiseSeparableConv(num_filters, num_filters, stride=1)
        self.ds12 = DepthwiseSeparableConv(num_filters, num_filters, stride=1)

        self.ds13 = DepthwiseSeparableConv(num_filters, num_filters * 2, stride=2)
        num_filters *= 2

        self.ds14 = DepthwiseSeparableConv(num_filters, num_filters, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(num_filters, num_classes)

    def forward(self, x):

        x = self.conv1(x)

        x = self.ds2(x)
        x = self.ds3(x)
        x = self.ds4(x)
        x = self.ds5(x)
        x = self.ds6(x)
        x = self.ds7(x)

        x = self.ds8(x)
        x = self.ds9(x)
        x = self.ds10(x)
        x = self.ds11(x)
        x = self.ds12(x)

        x = self.ds13(x)
        x = self.ds14(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x

def MobileNet(num_classes=10, num_filters=32, strideFistConv=1):
    return MobileNetV1(num_classes=num_classes, num_filters=num_filters, strideFistConv=strideFistConv)