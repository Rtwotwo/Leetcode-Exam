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
    """Layer Normalization基础的LayerNorm层实现
    normalized_shape (int或tuple): 需要归一化的维度形状
    eps (float): 用于数值稳定的小常数，防止除零错误
    elementwise_affine (bool): 是否使用可学习的缩放参数gamma/weight和偏移参数beta/bias"""
    def __init__(self,
        normalized_shape: Union[int, Tuple[int, ...]],
        eps: float = 1e-6,
        elementwise_affine: bool = True,
        **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        # 将 normalized_shape统一转为tuple格式
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        # 初始化可学习的参数：weight即gamma用于尺度缩放和bias即beta用于偏移
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        else:
            # 注册为None参数(PyTorch模块标准做法)
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 检查输入张量的最后若干维是否与 normalized_shape 匹配
        if x.shape[-len(self.normalized_shape):] != self.normalized_shape:
            raise ValueError(f'输入张量的最后维度{x.shape[-len(self.normalized_shape):]}与归一化形状{self.normalized_shape} 不匹配')
        ndim = x.ndim
        norm_dims = tuple(range(ndim - len(self.normalized_shape), ndim))
        # 计算均值和方差(注意LayerNorm使用有偏方差, 即unbiased=False)
        # 归一化: (x - mean) / sqrt(var + eps)
        mean = x.mean(dim=norm_dims, keepdim=True)
        var = x.var(dim=norm_dims, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        # 如果启用仿射变换, 则应用可学习的缩放和偏移
        if self.elementwise_affine:
            x_norm = self.weight * x_norm + self.bias
        return x_norm
    

class MLP(nn.Module):
    """使用1x1卷积的多层感知机MLP层的实现
    输入张量的维度和输出维度均为[B, C, H, W]"""
    def __init__(self, in_features:int,
                 hidden_features: Optional[int]=None,
                 out_features:Optional[int]=None,
                 act_layer:Type[nn.Module]=nn.GELU,
                 norm_layer:Type[nn.Module]=nn.BatchNorm2d,
                 drop:float=0.,
                 device=None,
                 dtype=None)->None:
        super().__init__() 
        # 当out_features和hidden_features为None或未被赋值时
        # 会自动使用in_features的值作为默认值
        dd = {"device":device, "dtype":dtype}
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.norm1 = norm_layer(in_features, **dd)
        # 设置基础两层全连接MLP
        self.fc1 = nn.Conv2d(in_features, hidden_features, **dd)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, **dd)
        self.drop = nn.Dropout(drop)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.norm1(x)
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class Block(nn.Module):
    """采用预归一化的Transformer-Block块的实现
        dim:输入通道数; num_heads:注意力头数;
        mlp_ratio:MLP 隐藏维度与嵌入维度的比率;
        qkv_bias:如果为True, 给查询、键、值添加一个可学习的偏置;
        qk_norm:如果为True, 对查询和键应用归一化;
        proj_bias:如果为True,给输出投影添加偏置;
        proj_drop:投影 dropout 率;
        attn_drop:注意力 dropout 率;
        init_values:层缩放的初始值;
        drop_path:随机深度率;act_layer:激活层;
        norm_layer:归一化层;mlp_layer:MLP 层"""
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
        super().__init__()
        