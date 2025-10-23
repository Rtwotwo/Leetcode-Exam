"""
Author: Redal
Date: 2025-10-14
Todo: vit.py for creating vision of transformer model
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import math
import torch
import torch.nn as nn
from typing import Union, Type, List, Dict, Any
from typing import Optional, Tuple, Callable, Set


class LayerNorm(nn.Module):
    """normalized_shape (int or tuple): 归一化的维度形状
    eps:数值稳定的小常数; elementwise_affine:是否使用可学习的gamma和beta"""
    def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], 
                 eps:float=1e-6,
                 elementsize_affine:bool=True,
                 **kwargs:Dict[str,Any])->None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementsize_affine = elementsize_affine
        # 初始化可学习的参数gamma尺度缩放和beta偏移因子
        if self.elementsize_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
            self.beta = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        else: 
            self.gamma = self.register_parameter('weight', None)
            self.beta = self.register_parameter('bias', None)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if x.shape[-len(self.normalized_shape):] != self.normalized_shape[-1]:
            raise ValueError(f'张量的形状与归一化形状不匹配')
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True)
        x_norm = (x - mean)/math.sqrt(var + self.eps)
        # 判断是否缩放和缩放
        if self.elementsize_affine:
            x_norm = self.gamma * x_norm + self.beta
        return x_norm
    

class MLP(nn.Module):
    def __init__(self, in_featuures:int,
                 hidden_features: Optional[int]=None,
                 out_features:Optional[int]=None,
                 act_layer:Type[nn.Module]=nn.GELU,
                 norm_layer:Type[nn.Module]=nn.BatchNorm2d,
                 drop:float=0.,
                 device=None,
                 dtype=None):
        super().__init__() 


class Block(nn.Module):
    def __init__(self, dim:int, 
                 num_heads:int, 
                 mip_ratio:float=4.0,
                 qkv_bias:bool=False, 
                 qk_norm:bool=False,
                 scale_attn_norm:bool=False,
                 scale_mlp_norm:bool=False, 
                 proj_bias:bool=True, 
                 proj_drop:float=0.,
                 attn_drop: float=0., 
                 init_values:Optional[float]=None,
                 drop_path: float=0.,
                 act_layer:Type[nn.Module]=nn.GELU,
                 norm_layer:Type[nn.Module]=LayerNorm,
                 mlp_layer: Type[nn.Module]=MLP,
                 device=None,
                 dtype=None)->None:
        """"""