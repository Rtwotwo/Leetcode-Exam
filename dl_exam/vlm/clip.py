"""
Author: Redal
Date: 2025-11-04
Todo: __init__.py for vlm tasks
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict
from typing import Tuple, Union


class Bottleneck(nn.Module):
    # 定义ResNet的BottleNeck结构的扩展系数
    expansion = 4
    def __init__(self, 
                 inplanes:int, 
                 planes:int, 
                 stride:int=1
                 )->None:
        # 所有卷积层的步进均为1
        # 当步进stride>1时, 会设置AvgPool2d层作为下采样
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(inplanes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)