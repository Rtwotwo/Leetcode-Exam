"""
Author: Redal
Date: 2025-10-14
Todo: resnet.py for creating ResNet model
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 预定义各类尺寸的ResNet版本, 包括18, 34, 50, 101, 152五种尺寸
__all__ = ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """定义3x3带padding的卷积层
    inplanes输入特征图的通道数
    out_planes输出特征图的通道数
    stride卷积核在输入特征图上滑动的步长
    groups控制输入和输出通道之间的连接方式,groups=1所有输入通道都与所有输出通道连接
    dilation空洞率/膨胀率用于空洞卷积,卷积核元素之间会有间隔,增大感受野而不增加参数量"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """定义1x1带padding的卷积层"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


# 定义ResNet基本卷积块
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, 
                 downsample=None, groups=1, base_width=64, 
                 dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None: norm_layer == nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError('Dilation > 1 not supported in BasicBlock')
        # 当步长(stride)不等于1时,self.conv1层和self.downsample层都会对输入进行下采样
        # 标准卷积+BN+激活层
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        # 初始化downsample和stride
        self.downsample = downsample
        self.stride = stride
    def forward(self, x):
        # 保存原始输入,用于后续的残差连接
        identity = x
        # 残差卷积块标准实现
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # 如果需要下采样(调整维度或步长),对输入进行处理
        if self.downsample is not None:
            identity  =self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


# ResNet中的瓶颈(Bottleneck)模块的实现
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1,
                 downsample=None, groups=1, base_width=64,
                 dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        