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


class LayerNorm(nn.LayerNorm):
    def __init__(self, normalized_shape:Tuple[int], 
                 eps:float=1e-6,
                 affine:bool=True,
                 **kwargs:Dict[str,Any])->None:
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        


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
