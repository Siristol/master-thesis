# MobileNetV1 implementation for ANN2SNN conversion
# Adapted for CIFAR datasets and ANN2SNN framework compatibility

from typing import Callable, Any, Optional

import torch
from torch import Tensor
from torch import nn

__all__ = [
    "MobileNetV1",
    "DepthWiseSeparableConv2d",
    "mobilenet_v1",
]


class MobileNetV1(nn.Module):

    def __init__(
            self,
            num_classes: int = 10, #Changed to match cifar10
    ) -> None:
        super(MobileNetV1, self).__init__()
        self.features = nn.Sequential(
            # First conv layer. Change from Conv2dNormActivation to Conv2d for better ANN2SNN conversion
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),  #Chaged stride to 1 for CIFAR datasets insteed of 2
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            DepthWiseSeparableConv2d(32, 64, 1),
            DepthWiseSeparableConv2d(64, 128, 2),
            DepthWiseSeparableConv2d(128, 128, 1),
            DepthWiseSeparableConv2d(128, 256, 2),
            DepthWiseSeparableConv2d(256, 256, 1),
            DepthWiseSeparableConv2d(256, 512, 2),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 512, 1),
            DepthWiseSeparableConv2d(512, 1024, 1),  # stride=1 to avoid too small feature maps insteed of 2
            DepthWiseSeparableConv2d(1024, 1024, 1),
        )

        # Adaptive average pooling for different input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) #Change from 7x7 to 1x1 for CIFAR datasets

        self.classifier = nn.Linear(1024, num_classes)

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        out = self._forward_impl(x)

        return out

    # Support torch.script function
    def _forward_impl(self, x: Tensor) -> Tensor:
        out = self.features(x)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1.0)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.zeros_(module.bias)


class DepthWiseSeparableConv2d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(DepthWiseSeparableConv2d, self).__init__()
        self.stride = stride
        if stride not in [1, 2]:
            raise ValueError(f"stride should be 1 or 2 instead of {stride}")

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv = nn.Sequential(
            # Depthwise convolution
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            norm_layer(in_channels),
            nn.ReLU(inplace=True),
            
            # Pointwise convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, 
                     padding=0, bias=False),
            norm_layer(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)

        return out


def mobilenet_v1(**kwargs: Any) -> MobileNetV1:
    model = MobileNetV1(**kwargs)

    return model