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

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(stride) if stride>1 else nn.Identity()

        self.conv3 = nn.Conv2d(inplanes, planes * self.axpa, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True) # 用于残差连接再激活
        # 初始化downsample和stride参数配置
        self.downsample = None
        self.stride = stride
        # 当条件成立时, 配置下采样层进行降采样, 用于残差连接的identity张量
        if self.stride>1 or inplanes!=planes*Bottleneck.expansion:
            self.downsample = nn.Sequential(OrderedDict([
                ('-1', nn.AvgPool2d(stride)),
                ('0', nn.Conv2d(inplanes, planes*self.expansion)),
                ('1', nn.BatchNorm2d(planes*self.expansion))]))
    def forward(self, x:torch.Tensor)->torch.Tensor:
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity =self.downsample(x)
        out += identity
        out = self.relu3(out)
        return out
    

class AttentionPool2d(nn.Module):
    """基于注意力机制的2D特征池化操作:
    通过自注意力机制,让类别token学习对所有空间位置特征的加权聚合,
    从而实现具有注意力机制的池化操作，比传统的平均/最大池化更具判别性
    spacial_dim: 输入特征图的空间维度,可能是H/W"""
    def __init__(self, 
                 spacial_dim:int,
                 embed_dim:int,
                 num_heads:int,
                 output_dim:int=None,
                 )->None:
        super().__init__()
        # spacial_dim²表示空间位置数量, +1通常用于添加一个特殊的类别token, embed_dim表示嵌入维度并作出归一化
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = x.flatten(start_dim=2).permute(2, 0, 1) # Adjust NCHW to (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0) # Add class token (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype) # Maintain shape (HW+1)NC
        # 直接调用torch提供的MHA的Forward函数
        # x[:1]是为了实现类别token对所有空间特征的注意力聚合操作
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_weight=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)
        return x.squeeze(0)
    

class ModifiedResNet(nn.Module):
    """与torchvision中的ResNet类相似但包含以下更改的ResNet类:
    现在有3个主干卷积,而非1个,并且使用平均池化而非最大池化,
    执行抗锯齿跨步卷积,其中在跨步大于1的卷积前添加一个平均池化,
    最终的池化层是QKV注意力机制AttentionPool2d,而非平均池化"""
    def __init__(self,
                 layers,
                 output_dim:int,
                 heads:int,
                 input_resolution:int=224,
                 width:int=64)->None:
        self.output_dim = output_dim
        self.input_resolution = input_resolution
        # 3层卷积层的主干卷积
        self.conv1 = nn.Conv2d(3, width//2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width//2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width//2, width//2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width//2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width//2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(kernel_size=2)

        # 残差模块层
        self._inplanes = width
        

