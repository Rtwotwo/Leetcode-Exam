"""
Author: Redal
Date: 2025-10-14
Todo: vit.py for creating vision of transformer model
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import os
import math
import torch
import torch.nn as nn
from functools import partial
from typing import Union, Type, List, Dict, Any
from typing import Optional, Tuple, Callable, Set


class DropPath(nn.Module):
    """随机丢弃路径的实现
    drop_prob(float): 丢弃概率"""
    def __init__(self, drop_prob:float=0.1,
                 eps:float=1e-6,
                 training:bool=True)->None:
        self.drop_prob = drop_prob
        self.eps = eps
        self.training = training
    def forward(self, x:torch.Tensor)->torch.Tensor:
        if not self.training or self.drop_prob==0.0:
            return x
        # 使用随机mask完成随机路径的筛选
        keep_prob = 1.0 -self.drop_prob
        mask = (torch.rand_like(x) < keep_prob).float(x) / keep_prob
        return x * mask


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


class PatchEmbed(nn.Module):
    """将输入图像划分为多个非重叠的patch并线性嵌入到高维空间
    img_size:输入图像的尺寸;
    patch_size:patch的尺寸;
    in_chans:输入图像的通道数;
    embed_dim:嵌入维度;
    norm_layer:归一化层;
    flatten:是否将patch展平为一维向量"""
    def __init__(self, img_size:int=224, 
                 patch_size:int=16, 
                 in_chans:int=3,
                 embed_dim:int=768)->None:
        super().__init__()
        # 计算patch数量
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        # 卷积层划分输入图像为patch
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x: torch.Tensor)->torch.Tensor:
        # 检查输入图像的尺寸是否与预设尺寸匹配
        B, C, H, W = x.shape
        assert H==self.img_size and W==self.img_size, \
            f'输入的图像尺寸{H}x{W}与预设尺寸{self.img_size}不匹配!!'
        # x:[B, C, H, W]->[B, embed_dim, H//p, W//p]->[B, embed_dim, H//p*W//p]
        # transpose(1,2)则是改变张量的维度顺序为[B, num_patches, embed_dim]
        x = self.proj(x).flatten(2).transpose(1,2)
        return x
    

class Attention(nn.Module):
    """Transformer的注意力机制实现
    dim:输入通道数; num_heads:注意力头数;
    qkv_bias:如果为True, 给查询、键、值添加一个可学习的偏置;
    qk_norm:如果为True, 对查询和键应用归一化;
    proj_bias:如果为True,给输出投影添加偏置"""
    def __init__(self, dim:int,
                 num_heads:int,
                 qkv_bias:bool=False,
                 qk_norm:bool=False,
                 attn_drop:float=0.,
                 proj_drop:float=0.,
                 norm_layer:Type[nn.Module]=LayerNorm)->None:
        super().__init__()
        # 计算每个注意力头的维度
        assert dim%num_heads==0, f'输入通道数{dim}不能被注意力头数{num_heads}整除!!'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # 初始化查询、键、值映射层
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # 输入张量的维度为[B, N, C]即[B, num_patches, embed_dim]
        B, N, C = x.shape
        # [B, N, C]->[B, N, 3*C]->[B, N, 3, num_heads, head_dim]->[3, B, num_heads, N, head_dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0) # 张量按照第 0 维度(最外层维度)进行拆分
        q, k = self.q_norm(q), self.k_norm(k)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        # 乘以值并求和输出得到注意力的实际得分
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
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
                 mlp_ratio:float=4.0,
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
        # 设置Transformer-Block块,主要涵盖Encoder部分
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, qk_norm,
                            attn_drop, proj_drop, norm_layer)
        # LayerScale对神经网络层进行自适应缩放
        self.ls1 = nn.Parameter(init_values*torch.ones(dim)) if init_values else None
        self.drop_path1 = DropPath(drop_path) if drop_path>0. else nn.Indentity()
        self.norm2 = norm_layer(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio), act_layer=act_layer)
        self.ls2 = nn.Parameter(init_values*torch.ones(dim)) if init_values else None
        self.drop_path2 = DropPath(drop_path) if drop_path>0. else nn.Identity()
    def forward(self, x:torch.Tensor)->torch.Tensor:
        # 注意力机制层同时决定是否使用LayerScale
        if self.ls1 is not None:
            x = x + self.drop_path1(self.ls1 * self.attn(self.norm1(x)))
        else: x = x + self.drop_path1(self.attn(self.norm1(x)))
        # MLP作为VIT-Block块的前馈层并使用LayerScale
        if self.ls2 is not None:
            x = x +self.drop_path2(self.ls2*self.mlp(self.norm2(x)))
        else: x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x
    

class VisionTransformer(nn.Module):
    """Vision Transformer的实现
    img_size:输入图像尺寸; patch_size:patch尺寸; in_chans:输入通道数;
    embed_dim:嵌入维度; depth:编码器块数; num_heads:注意力头数;
    mlp_ratio:MLP  hidden_dim/embed_dim; qkv_bias:给查询、键、值添加一个可学习的偏置;
    qk_norm:对查询和键应用归一化; proj_bias:给输出投影添加偏置; proj_drop:投影 dropout 率;
    attn_drop:注意力 dropout 率; init_values:层缩放的初始值; drop_path:随机深度率;
    act_layer:激活层; norm_layer:归一化层; mlp_layer:MLP 层; device:设备; dtype:数据类型"""
    