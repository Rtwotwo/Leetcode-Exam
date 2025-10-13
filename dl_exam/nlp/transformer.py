"""
Author: Redal
Date: 2025-10-11
Todo: transformer.py
Homepape: https://github.com/Rtwotwo/Code-Exam.git
"""
import math
import torch
import torch.nn as nn


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


def dropout(x, p=0.1, training=True):
    if not training or p==0.0:
        return x
    keep_prob = 1-p
    mask = (torch.rand_like(x) < keep_prob).float() / keep_prob
    return x * mask


def gelu(x):
    # GELU(x) = x·Φ(x), 其中Φ(x)是标准正态分布的累积分布函数(CDF)
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
        

# 创建transformer架构组件
class MultiHeadAttention:
    def __init__(self, d_model, num_heads, dropout=0.1):
        assert d_model % num_heads == 0, "d_model nust be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.dropout_p = dropout
        # 创建q, k, v, out权重矩阵
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.out_proj = Linear(d_model, d_model)
        # 创建缩放因子, 使用head_dim控制缩放
        self.scale = math.sqrt(self.head_dim)
    def __call__(self, query, key, value, mask=None, training=True):
        # query, key, value形状均为[batch_size, seq_len, d_model]
        # batch_size批次大小, seq_len序列长度, d_model模型特征维度
        batch_size = query.shape[0]
        seq_len_q = query.shape[1]
        seq_len_k = key.shape[1]
        # 计算Q, K, V线性投影, 使用Linear层处理
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        # 参数空间映射到更适合注意力计算的空间
        # 让每个注意力头关注不同的表示子空间, 注意V同样使用seq_len_k
        Q = Q.view(batch_size, seq_len_q, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_k, self.num_heads, self.head_dim).transpose(1, 2)
        # 计算注意力得分，使用Q, K进行矩阵乘法 atten_scores = (Q @ K.t) / sqrt(d)
        # 除以scale（√head_dim）进行缩放，防止点积过大导致梯度问题
        atten_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        if mask is not None:
            # 支持 bool mask 或 0/1 mask
            if mask.dtype == torch.bool:
                atten_scores = atten_scores.masked_fill(~mask, float('-inf'))
            else:
                atten_scores = atten_scores.masked_fill(mask==0, float('-inf'))
        
        # 计算注意力权重 atten_weights = softmax(atten_scores)
        # 计算注意力输出 atten_output = atten_weights @ V
        atten_weights = torch.softmax(atten_scores, dim=-1)
        atten_weights = dropout(atten_weights, p=self.dropout_p, training=training)
        atten_output = torch.matmul(atten_weights, V)
        # 将多头注意力输出拼接在一起, 并进行线性变换
        atten_output = atten_output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, self.d_model)
        output = self.out_proj(atten_output)
        return output


class PositionalEncoding:
    def __init__(self, d_model, max_len=512):
        # 固定正弦位置编码是预先计算好、不可训练的, 效果有限
        # 使用可训练的pos_embedding, 可以在训练过程中更新位置编码
        self.pos_embedding = torch.nn.Parameter(torch.zeros(1, max_len, d_model))
    def __call__(self, x):
        seq_len = x.shape[1]
        return x + self.pos_embedding[:, :seq_len, :]


class FeedForward:
    def __init__(self, d_model, dff, dropout=0.1):
        self.w1 = Linear(d_model, dff)
        self.w2 = Linear(dff, d_model)
        self.dropout_p = dropout
    def __call__(self, x, training=True):
        x = self.w1(x)
        x = gelu(x)
        x = dropout(x, p=self.dropout_p, training=training)
        x = self.w2(x)
        return x


