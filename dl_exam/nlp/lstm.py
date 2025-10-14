"""
Author: Redal
Date: 2025-10-14
Todo: lstm.py for creating lstm and GRU model
Homepage: https://github.com/Rtwotwo/Code-Exam.git
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# 标准LSTM与PyTorch nn.LSTM功能一致,用于教学/自定义扩展
# 包含: 遗忘门, 输入门, 输出门+记忆单元(Cell State); 相关公式包括: 
# f_t = \sigma(W_{xf}·x_t + W_{hf}·h_{t-1} + b_f)
# i_t &= \sigma(W_{xi}·x_t + W_{hi}·h_{t-1} + b_i) 
# \tilde{C}_t &= \tanh(W_{xc}·x_t + W_{hc}·h_{t-1} + b_c) 
# C_t &= f_t \odot C_{t-1} + i_t \odot \tilde{C}_t 
# o_t &= \sigma(W_{xo}·x_t + W_{ho}·h_{t-1} + b_o) 
# h_t &= o_t \odot \tanh(C_t)
class VanillaLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, 
                 num_layers=1, bias=True, dropout=0.0, 
                 bidirectional=False):
        super().__init__()
        # 初始化LSTM层所需超参数
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # 使用PyTorch内置LSTM(高效且支持cuDNN)
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bias=self.bias,
                            dropout=self.dropout,
                            bidirectional=self.bidirectional)
    def forward(self, x, h0=None, c0=None):
        batch_size = x.size(0)
        if h0 is None:
            h0 = torch.zeros(self.num_layers * self.num_directions, batch_size, 
                             self.hidden_size, device=x.device)
        if c0 is None:
            c0 = torch.zeros(self.num_layers * self.num_directions, batch_size,
                             self.hidden_size, device=x.device)
        output, (hn, cn) = self.lstm(x, (h0, c0))
        return output, (hn, cn)

        