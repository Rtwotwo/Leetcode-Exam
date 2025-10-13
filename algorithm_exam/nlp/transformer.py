"""
Author: Redal
Date: 2025-10-11
Todo: transformer.py
Homepape: https://github.com/Rtwotwo/Code-Exam.git
"""
import math
import torch
import torch.nn as nn


def dropout(x, p=0.1, training=True):
    if not training or p==0.0:
        return x
    keep_prob = 1-p
    mask = (torch.rand_like(x) < keep_prob).float() / keep_prob
    return x * mask


def gelu(x):
    #  GELU(x) = x·Φ(x), 其中Φ(x)是标准正态分布的累积分布函数(CDF)
    # GELU(x) ≈ 0.5x · (1 + tanh(√(2/π) · (x + 0.044715x³)))
    return x * 0.5 * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class LayerNorm:
    def __init__(self, normalized_shape, eps=1e-5):
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape, )
        self.normalized_shape = normalized_shape
        self.eps = eps
        # 必须初始化参数为Parameter
        self.weight = torch.nn.Parameter(torch.ones(normalized_shape))
        self.bias = torch.nn.Parameter(torch.zeros(normalized_shape))
    def __call__(self, x):
        # x: (..., *normalized_shape)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.weight * x_norm + self.bias
    

class Linear:
    def __init__(self, in_features, out_features, bias=True):
        self.in_features = in_features
        self.out_features = out_features
        # 使用Xavier近似初始化
        bound = 1 / math.sqrt(in_features)
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features, in_features).uniform_(-bound, bound))
        else:
            self.bias = None
    def __call__(self, x):
        # x: (*, in_features) -> (*, out_features)
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output += self.bias
        return output
        

# 设置transformer不同尺寸的经验性设定模型参数
MODEL_CONFIGS = {
    'tiny': {
        'num_layers': 2,
        'd_model': 128,
        'num_heads': 2,
        'dff': 512,
        'vocab_size': 30522,
        'max_seq_len': 512,
        'dropout': 0.1},
    'small': {
        'num_layers': 4,
        'd_model': 256,
        'num_heads': 4,
        'dff': 1024,
        'vocab_size': 30522,
        'max_seq_len': 512,
        'dropout': 0.1},
    'medium': {
        'num_layers': 6,
        'd_model': 512,
        'num_heads': 8,
        'dff': 2048,
        "vocab_size": 30522,
        'max_seq_len': 512,
        'dropout': 0.1},
    'large': {
        'num_layers': 12,
        'd_model': 768,
        'num_heads': 12,
        'dff': 3072,
        'vocab_size': 30522,
        'max_seq_len': 512,
        'dropout': 0.1},
    'huge': {
        'num_layers': 24,
        'd_model': 1024,
        'num_heads': 16,
        'dff': 4096,
        'vocab_size': 30522,
        'max_seq_len': 512,
        'dropout': 0.1},}
